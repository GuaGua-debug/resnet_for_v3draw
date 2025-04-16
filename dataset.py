import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import struct
import numpy as np

def load_v3d_raw_img_file1(filename):
    im = {}

    f_obj = open(filename, 'rb')

    # read image header - formatkey(24bytes)
    len_formatkey = len('raw_image_stack_by_hpeng')
    formatkey = f_obj.read(len_formatkey)
    formatkey = struct.unpack(str(len_formatkey) + 's', formatkey)
    if formatkey[0] != b'raw_image_stack_by_hpeng':
        print("ERROR: File unrecognized (not raw, v3draw) or corrupted.")
        f_obj.close()
        return im

    # read image header - endianCode(1byte)
    endiancode = f_obj.read(1)
    endiancode = struct.unpack('c', endiancode)  # 'c' = char
    endiancode = endiancode[0]
    if endiancode != b'B' and endiancode != b'L':
        print("ERROR: Only supports big- or little- endian,"
              " but not other format. Check your data endian.")
        f_obj.close()
        return im

    # read image header - datatype(2bytes)
    datatype = f_obj.read(2)
    if endiancode == b'L':
        datatype = struct.unpack('<h', datatype)  # 'h' = short
    else:
        datatype = struct.unpack('>h', datatype)  # 'h' = short
    datatype = datatype[0]
    if datatype < 1 or datatype > 4:
        print("ERROR: Unrecognized data type code [%d]. "
              "The file type is incorrect or this code is not supported." % (datatype))
        f_obj.close()
        return im

    # read image header - size(4*4bytes)
    size = f_obj.read(4 * 4)
    if endiancode == b'L':
        size = struct.unpack('<4l', size)  # 'l' = long
    else:
        size = struct.unpack('>4l', size)  # 'l' = long
    # print(size)

    # read image data
    npixels = size[0] * size[1] * size[2] * size[3]
    im_data = f_obj.read()
    if datatype == 1:
        im_data = np.frombuffer(im_data, np.uint8)
    elif datatype == 2:
        im_data = np.frombuffer(im_data, np.uint16)
    else:
        im_data = np.frombuffer(im_data, np.float32)
    if len(im_data) != npixels:
        print("ERROR: Read image data size != image size. Check your data.")
        return im


    im_data = im_data.reshape((size[3], size[2], size[1], size[0]))
    # print(im_data.shape)
    # print(im_data.shape)
    im_data = np.moveaxis(im_data, 0, -1)
    # print(im_data.shape)
    im_data = np.moveaxis(im_data, 0, -2)
    # print(im_data.shape)
    f_obj.close()

    im['endian'] = endiancode
    im['datatype'] = datatype
    im['size'] = im_data.shape
    im['data'] = im_data

    return im

'''
dataset
'''
class V3drawDataset(Dataset):

    def __init__(self, images_path, labels_path, transform=None):

        self.folder_path = images_path
        self.transform = transform

        name_list = ['num', 'filename', 'type', 'name', 'label', 'blank', 'date']
        self.labels_df = pd.read_csv(labels_path, names=name_list)
        self.file_list = self.labels_df['filename'].tolist()
        self.labels = self.labels_df['label'].tolist()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        file_path = os.path.join(self.folder_path, filename)
        image = load_v3d_raw_img_file1(file_path)
        if not image:
            raise ValueError(f"Failed to load image from {file_path}")

        image_data = image['data']
        image_data = image_data.astype(np.float32)

        label = self.labels[idx]
        label = torch.tensor(0) if label[:1] == '4' else torch.tensor(1)

        if self.transform:
            image_data = self.transform(image_data)

        return image_data, label

    def get_sampler(self):
        class_counts = [0, 0]
        for label in self.labels:
            label_idx = 0 if label[:1] == '4' else 1
            class_counts[label_idx] += 1

        weights = []
        for label in self.labels:
            label_idx = 0 if label[:1] == '4' else 1
            weight = 1.0 / class_counts[label_idx]
            weights.append(weight)

        weights = torch.DoubleTensor(weights)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )

        return sampler


'''
数据Transform
'''
class Transform4D:
    '''输入图像：原始numpy(float32)，四维(H, W, D, C)
    输出图像：tensor张量，归一化，四维(C, H, W, D)'''
    '''numpy(256, 256, 64, 1) ——> tensor(1, 256, 256, 64)'''

    def __init__(self, totensor=True, permute=True, normalize=True):
        self.totensor = totensor
        self.permute = permute
        self.normalize = normalize

    def __call__(self, x):
        if self.totensor:
            x = torch.from_numpy(x)

        if self.permute:
            x = x.permute(3, 0, 1, 2)

        if self.normalize and x.max() > 1.0:
            x = x/255.0

        return x

class Augmentation3D:
    def __init__(self, flip_prob=0.5, rotate_prob=0.5, noise_prob=0.3):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.noise_prob = noise_prob

    def __call__(self, x):
        '''输入数据（经过Transform4D）:四维tensor(C, H, W, D)
        输出数据：随机增强后图像'''

        # 随机翻转
        if torch.rand(1).item() < self.flip_prob:
            dim = torch.randint(1, 4, (1,)).item()
            x = torch.flip(x, [dim])

        # 随机旋转
        if torch.rand(1).item() < self.rotate_prob:
            k = torch.randint(1, 4, (1,)).item()
            x = torch.rot90(x, k, dims=[1, 2])

        # 高斯噪声
        if torch.rand(1).item() < self.noise_prob:
            noise = torch.randn_like(x) * 0.03
            x = x + noise
            x = torch.clamp(x, 0, 1)
        return x

class Gamma_correction:
    '''
    对比度增强
    '''

    def __init__(self, gamma = 2.2):
        self.gamma = gamma

    def __call__(self, x):
        normalized = x / 255.0
        corrected = np.power(normalized, self.gamma)
        result = corrected * 255

        return result