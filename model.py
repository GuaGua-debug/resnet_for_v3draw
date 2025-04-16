import torch.nn as nn

'''
model
'''
class ConvBlock3D(nn.Module):
    # 三维卷积（k = 3）

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class SEBlock3D(nn.Module):

    def __init__(self, channel, reduction=16):
        super(SEBlock3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock3D(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, se=True):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = ConvBlock3D(in_channels, out_channels, stride=stride)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # SE注意力模块
        self.se = se
        if se:
            self.se_block = SEBlock3D(out_channels)

        # 残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)

        # 应用SE注意力
        if self.se:
            out = self.se_block(out)

        # 添加残差连接
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet_3d_binary(nn.Module):

    def __init__(self, in_channels=1, num_classes=2, dropout_rate=0.3):
        super(ResNet_3d_binary, self).__init__()

        # 初始卷积层
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        # 残差块
        self.layer1 = nn.Sequential(
            ResidualBlock3D(16, 32, stride=1, se=True),
            ResidualBlock3D(32, 32, stride=1, se=True)
        )

        self.layer2 = nn.Sequential(
            ResidualBlock3D(32, 64, stride=2, se=True),
            ResidualBlock3D(64, 64, stride=1, se=True)
        )

        self.layer3 = nn.Sequential(
            ResidualBlock3D(64, 128, stride=2, se=True),
            ResidualBlock3D(128, 128, stride=1, se=True)
        )

        self.layer4 = nn.Sequential(
            ResidualBlock3D(128, 256, stride=2, se=True)
        )

        # 分类层
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout_rate)  # dropout减小过拟合
        self.fc = nn.Linear(256, num_classes)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x