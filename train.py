import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

from dataset import V3drawDataset, Transform4D, Augmentation3D, Gamma_correction
from model import ResNet_3d_binary


def draw_fig(list, epoch, phase, type, color):
    x1 = np.arange(0, epoch+1, 1)
    y1 = list
    plt.cla()
    plt.grid()
    plt.title(label = phase + type + ' vs. epoch', fontsize=20)
    plt.plot(x1, y1, '.-', color = color)
    plt.xlabel('epoch', fontsize=20)
    plt.ylabel(type, fontsize=20)
    if type == 'ACC':
        plt.ylim((0, 1))
    plt.savefig('' + phase + '_' + type + '.png')

'''
train
'''
def train_model(model, train_loader, val_loader, num_epochs=30, lr=0.001):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, patience=4, verbose=True)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    train_losses = []
    train_acces = []
    val_losses = []
    val_acces = []
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # 训练
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f'Epoch {epoch + 1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')
        train_losses.append(float(format(epoch_loss, '.3f')))
        train_acces.append(float(format(epoch_acc, '.3f')))
        logging.info(f'Epoch {epoch + 1}/{num_epochs} - Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

        # 验证
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data).item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects / len(val_loader.dataset)

        print(f'Epoch {epoch + 1}/{num_epochs} - Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}')
        val_losses.append(float(format(val_loss, '.3f')))
        val_acces.append(float(format(val_acc, '.3f')))
        logging.info(f'Epoch {epoch + 1}/{num_epochs} - Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}')

        # save figs
        draw_fig(train_losses, int(epoch), "Training" ,"Loss", "blue")
        draw_fig(train_acces, int(epoch), "Training" ,"ACC", "orange")
        draw_fig(val_losses, int(epoch), "Validating" ,"Loss", "red")
        draw_fig(val_acces, int(epoch), "Validating" ,"ACC", "green")

        # 更新学习率
        scheduler.step(val_loss)
        # scheduler.step()

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_cell_classifier.pth')
            print(f'Model saved with accuracy: {val_acc:.4f}')
            logging.info(f'Model saved with accuracy: {val_acc:.4f}')

    return model


if __name__ == "__main__":
    logging.basicConfig(filename='log.txt', filemode='w',
                        format='%(asctime)s - %(message)s', level=logging.INFO)
    image_path = r'data'
    train_csv_path = r'label/train.csv'
    val_cav_path = r'label/val.csv'

    # 创建模型
    model = ResNet_3d_binary(in_channels=1, num_classes=2)

    # 打印模型
    print(model)

    # test数据
    # x = torch.randn(1, 1, 256, 256, 64)
    # output = model(x)
    # print(f"Output shape: {output.shape}")

    transform = torchvision.transforms.Compose([
        Gamma_correction(2.2),
        Transform4D(),
        Augmentation3D()
    ])

    training_dataset = V3drawDataset(images_path=image_path, labels_path=train_csv_path, transform=transform)
    validating_dataset = V3drawDataset(images_path=image_path, labels_path=val_cav_path, transform=transform)

    # 随机采样数据
    train_sampler = training_dataset.get_sampler()
    train_loader = DataLoader(training_dataset, batch_size=24, sampler=train_sampler, num_workers=8, drop_last=False)
    val_loader = DataLoader(validating_dataset, batch_size=24, shuffle=False, num_workers=8, drop_last=False)

    print(f"训练数据集大小: {len(training_dataset)}")
    print(f"验证数据集大小: {len(validating_dataset)}")
    print(f"训练批次数量: {len(train_loader)}")
    print(f"验证批次数量: {len(val_loader)}")
    logging.info(f"Training Dataset: {len(training_dataset)}")
    logging.info(f"Validating Dataset: {len(validating_dataset)}")
    logging.info(f"Training batches: {len(train_loader)}")
    logging.info(f"Validating batches: {len(val_loader)}")

    # 开始训练
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=200,
        lr=0.001
    )

    print("train comnplete!")
    print(trained_model)
