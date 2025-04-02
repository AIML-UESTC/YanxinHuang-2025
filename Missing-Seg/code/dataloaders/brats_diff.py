import os

import pywt
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler
import nibabel as nib
# import numpy as np
import SimpleITK as sitk
class RandomCrop_multi(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        # image, label = sample['T1'], sample['label']
        T1,T1ce,T2w,T2f, label = sample['T1'],sample['T1ce'],sample['T2w'],sample['T2f'], sample['label']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            T1 = np.pad(T1, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            T1ce = np.pad(T1ce, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            T2w = np.pad(T2w, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            T2f = np.pad(T2f, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)],
                             mode='constant', constant_values=0)

        (w, h, d) = T1.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        T1 = T1[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        T1ce = T1ce[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        T2w = T2w[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        T2f = T2f[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
            # return {'image': image, 'label': label, 'sdf': sdf}
            return {'T1':T1,'T1ce':T1ce,'T2w':T2w,'T2f':T2f, 'label': label,'sdf':sdf}
        else:
            return {'T1':T1,'T1ce':T1ce,'T2w':T2w,'T2f':T2f, 'label': label}
class RandomRotFlip_multi(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        # image, label = sample['image'], sample['label']
        T1,T1ce,T2w,T2f, label = sample['T1'],sample['T1ce'],sample['T2w'],sample['T2f'], sample['label']
        k = np.random.randint(0, 4)
        T1 = np.rot90(T1, k)
        T1ce = np.rot90(T1ce, k)
        T2w = np.rot90(T2w, k)
        T2f = np.rot90(T2f, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        T1 = np.flip(T1, axis=axis).copy()
        T1ce = np.flip(T1ce, axis=axis).copy()
        T2w = np.flip(T2w, axis=axis).copy()
        T2f = np.flip(T2f, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        # return {'image': image, 'label': label}
        return {'T1':T1,'T1ce':T1ce,'T2w':T2w,'T2f':T2f, 'label': label}

def DWT(img):
    # 进行小波变换
    coeffs = pywt.dwtn(img, 'db1')  # Daubechies 1 小波
    cA = coeffs['aaa']  # 低频分量

    # 创建一个包含低频分量的系数字典，其他系数设为零
    zero_coeffs = {key: np.zeros_like(coeffs[key]) for key in coeffs}
    zero_coeffs['aaa'] = cA

    # 进行逆变换，得到q_cA
    q_cA = pywt.idwtn(zero_coeffs, 'db1')
    q_cA = q_cA[:, :, :155]

    # 定义容差
    tolerance = 1e-9

    # # 使用容差进行比较，代替 q_cA == 1 等判断
    # q_cA = [(np.isclose(q_cA, 1, atol=tolerance)) | (np.isclose(q_cA, 3, atol=tolerance)),
    #         (np.isclose(q_cA, 1, atol=tolerance)) | (np.isclose(q_cA, 3, atol=tolerance)) | (
    #             np.isclose(q_cA, 2, atol=tolerance)),
    #         np.isclose(q_cA, 3, atol=tolerance)]
    q_cA = np.stack(q_cA, axis=0).astype(np.float32)
    high_freq_coeffs = {key: coeffs[key] for key in coeffs if key != 'aaa'}

    # 对每个高频分量进行小波逆变换
    high_freq_images = []
    for key, value in high_freq_coeffs.items():
        zero_coeffs = {k: np.zeros_like(coeffs[k]) for k in coeffs}
        zero_coeffs[key] = value  # 只保留该高频分量
        high_freq_image = pywt.idwtn(zero_coeffs, 'db1')  # 逆变换
        high_freq_images.append(np.expand_dims(high_freq_image[:,:,:155],axis = 0))
    cat_high_freq = np.concatenate(high_freq_images, axis=0)
    return q_cA,cat_high_freq
def normalize(mask):
    min_val = np.min(mask)
    max_val = np.max(mask)
    normalized_mask = (mask - min_val) / (max_val - min_val)
    return normalized_mask

def quantize(mask, num_classes=4):
    # 归一化掩膜
    normalized_mask = normalize(mask)
    # 量化到 [0, 1, 2, 3] 四个值
    quantized_mask = (normalized_mask * (num_classes - 1)).astype(int)
    return quantized_mask

# 示例
mask = np.array([[0.1, 0.5, 0.9], [0.2, 0.6, 0.8]])
class BraTS2019(Dataset):
    """ BraTS2019 Dataset """

    def __init__(self, base_dir=None, split='train', num=None, transform=None,mod = None,sup = None,transform_2=None):
        self._base_dir = base_dir
        self.transform = transform
        # self.transform_2 = transform_2
        self.sample_list = []
        self.mod = mod

        train_path = self._base_dir+'/train.txt'
        test_path = self._base_dir+'/val.txt'

        if split == 'train':
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()
        if sup == 'sup':
            self.image_list = [item.replace('\n', '').split(",")[0] for item in self.image_list][:100]
        else :
            self.image_list = [item.replace('\n', '').split(",")[0] for item in self.image_list]
        # self.image_list = [item.replace('\n', '').split(",")[0] for item in self.image_list]
        self.full = self.image_list[:100]
        self.semi = self.image_list[100:]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        try:
            image_name = self.image_list[idx]
            # image_name = self.full[idx]
            label_name = self.image_list[idx].replace(image_name[:-7].split('-')[-1], 'seg')
            image_path = os.path.join(self._base_dir, f"{image_name}")
            label_path = os.path.join(self._base_dir,  f"{label_name}")
            T1 = nib.load(image_path).get_fdata()
            T1ce = nib.load(image_path.replace(image_name[:-7].split('-')[-1], 't1c')).get_fdata()
            T2w = nib.load(image_path.replace(image_name[:-7].split('-')[-1], 't2w')).get_fdata()
            T2f = nib.load(image_path.replace(image_name[:-7].split('-')[-1], 't2f')).get_fdata()
            label = nib.load(label_path).get_fdata()
            sample_1 = {'T1': T1, 'T1ce': T1ce, 'T2w': T2w, 'T2f': T2f, 'label': label.astype(np.uint8)}
            if not os.path.exists(image_path) or not os.path.exists(label_path):
                raise FileNotFoundError(f"File {image_path} or {label_path} not found")
            if self.transform:
                sample = self.transform(sample_1)

        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            raise
        image = torch.cat((sample['T1'], sample['T1ce'], sample['T2w'], sample['T2f']), dim=0)
        sample = {'image': image, 'label': sample['label']}
        seg_data = sample['label']
        result = [(seg_data == 1) | (seg_data == 3), (seg_data == 1) | (seg_data == 3) | (seg_data == 2), seg_data == 3]
        reshape = np.stack(result, axis=0).astype(np.float32)
        sample = {'image': image, 'label': reshape,'image_name':image_name,'rere':sample['label']}
        return sample

class BraTS2019_consis(Dataset):
    """ BraTS2019 Dataset """

    def __init__(self, base_dir=None, split='train', num=None, transform=None,mod = None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        self.mod = mod

        train_path = self._base_dir+'/train.txt'
        test_path = self._base_dir+'/val.txt'

        if split == 'train':
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '').split(",")[0] for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        label_name = self.image_list[idx].replace(image_name[:-7].split('-')[-1], 'seg')
        if self.mod =='multi_concat':
            image_path = os.path.join(self._base_dir, f"{image_name}")
            label_path = os.path.join(self._base_dir,  f"{label_name}")
            T1 = nib.load(image_path).get_fdata()
            T1ce = nib.load(image_path.replace(image_name[:-7].split('-')[-1], 't1c')).get_fdata()
            T2w = nib.load(image_path.replace(image_name[:-7].split('-')[-1], 't2w')).get_fdata()
            T2f = nib.load(image_path.replace(image_name[:-7].split('-')[-1], 't2f')).get_fdata()
            # image = 0.25*T1+0.25*T1ce+0.25*T2w+0.25*T2f
            label = nib.load(label_path).get_fdata()
            label[label != 0] = 1
            if not os.path.exists(image_path) or not os.path.exists(label_path):
                raise FileNotFoundError(f"File {image_path} or {label_path} not found")
            sample_1 = {'T1':T1,'T1ce':T1ce,'T2w':T2w,'T2f':T2f, 'label': label.astype(np.uint8)}
            if self.transform:
                sample_1 = self.transform(sample_1)
            image = torch.cat((sample_1['T1'],sample_1['T1ce'],sample_1['T2w'],sample_1['T2f']), dim=0)
            noise = torch.randn_like(image)
            noise_image = image+noise
            sample = {'image': image, 'label': sample_1['label'],'Noise':noise_image}

        return sample


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)],
                             mode='constant', constant_values=0)

        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'sdf': sdf}
        else:
            return {'image': image, 'label': label}

class RandomCrop_label(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        # image, label = sample['T1'], sample['label']
        # T1,T1ce,T2w,T2f, label = sample['T1'],sample['T1ce'],sample['T2w'],sample['T2f'], sample['label']
        label = sample
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            # T1 = np.pad(T1, [(pw, pw), (ph, ph), (pd, pd)],
            #                mode='constant', constant_values=0)
            # T1ce = np.pad(T1ce, [(pw, pw), (ph, ph), (pd, pd)],
            #                mode='constant', constant_values=0)
            # T2w = np.pad(T2w, [(pw, pw), (ph, ph), (pd, pd)],
            #                mode='constant', constant_values=0)
            # T2f = np.pad(T2f, [(pw, pw), (ph, ph), (pd, pd)],
            #                mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)],
                             mode='constant', constant_values=0)

        # (w, h, d) = T1.shape
        (w, h, d) = label.shape

        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        # T1 = T1[w1:w1 + self.output_size[0], h1:h1 +
        #               self.output_size[1], d1:d1 + self.output_size[2]]
        # T1ce = T1ce[w1:w1 + self.output_size[0], h1:h1 +
        #               self.output_size[1], d1:d1 + self.output_size[2]]
        # T2w = T2w[w1:w1 + self.output_size[0], h1:h1 +
        #               self.output_size[1], d1:d1 + self.output_size[2]]
        # T2f = T2f[w1:w1 + self.output_size[0], h1:h1 +
        #               self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
            # return {'image': image, 'label': label, 'sdf': sdf}
            # return {'T1':T1,'T1ce':T1ce,'T2w':T2w,'T2f':T2f, 'label': label,'sdf':sdf}
            return label
        else:
            return label
            # return {'T1':T1,'T1ce':T1ce,'T2w':T2w,'T2f':T2f, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label}

class RandomRotFlip_label(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        # image, label = sample['image'], sample['label']
        # T1,T1ce,T2w,T2f, label = sample['T1'],sample['T1ce'],sample['T2w'],sample['T2f'], sample['label']
        label = sample
        k = np.random.randint(0, 4)
        # T1 = np.rot90(T1, k)
        # T1ce = np.rot90(T1ce, k)
        # T2w = np.rot90(T2w, k)
        # T2f = np.rot90(T2f, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        # T1 = np.flip(T1, axis=axis).copy()
        # T1ce = np.flip(T1ce, axis=axis).copy()
        # T2w = np.flip(T2w, axis=axis).copy()
        # T2f = np.flip(T2f, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        # return {'image': image, 'label': label}
        # return {'T1':T1,'T1ce':T1ce,'T2w':T2w,'T2f':T2f, 'label': label}
        return  label


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(
            image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros(
            (self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label, 'onehot_label': onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(
            1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        # label = sample['label']
        # label = label.reshape(
        #     1, label.shape[0], label.shape[1], label.shape[2])

        # image = image.reshape(
        #     image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}
            # return {'image': torch.from_numpy(image), 'label': torch.from_numpy(label).long()}

class ToTensor_multi(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # image = sample['image']
        # T1,T1ce,T2w,T2f = sample['T1'],sample['T1ce'],sample['T2w'],sample['T2f']
        # T1 = T1.reshape(
        #     1, T1.shape[0], T1.shape[1], T1.shape[2]).astype(np.float32)
        # T1ce = T1ce.reshape(
        #     1, T1ce.shape[0], T1ce.shape[1], T1ce.shape[2]).astype(np.float32)
        # T2w = T2w.reshape(
        #     1, T2w.shape[0], T2w.shape[1], T2w.shape[2]).astype(np.float32)
        # T2f = T2f.reshape(
        #     1, T2f.shape[0], T2f.shape[1], T2f.shape[2]).astype(np.float32)
        # if 'onehot_label' in sample:
        #     return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
        #             'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        #
        #     return {'T1':torch.from_numpy(T1),'T1ce':torch.from_numpy(T1ce),'T2w':torch.from_numpy(T2w),'T2f':torch.from_numpy(T2f), 'label': torch.from_numpy(sample['label']).long(),
        #             'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        # else:
        #     return {'T1':torch.from_numpy(T1),'T1ce':torch.from_numpy(T1ce),'T2w':torch.from_numpy(T2w),'T2f':torch.from_numpy(T2f), 'label': torch.from_numpy(sample['label']).long()}
        #     return {'image': torch.from_numpy(image), 'label': torch.from_numpy(label).long()}
        # return torch.from_numpy(sample).long()
        T1, T1ce, T2w, T2f = sample['T1'], sample['T1ce'], sample['T2w'], sample['T2f']
        T1 = T1.reshape(
            1, T1.shape[0], T1.shape[1], T1.shape[2]).astype(np.float32)
        T1ce = T1ce.reshape(
            1, T1ce.shape[0], T1ce.shape[1], T1ce.shape[2]).astype(np.float32)
        T2w = T2w.reshape(
            1, T2w.shape[0], T2w.shape[1], T2w.shape[2]).astype(np.float32)
        T2f = T2f.reshape(
            1, T2f.shape[0], T2f.shape[1], T2f.shape[2]).astype(np.float32)
        # label = sample['label']
        # label = label.reshape(
        #     1, label.shape[0], label.shape[1], label.shape[2])

        # image = image.reshape(
        #     image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            # return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
            #         'onehot_label': torch.from_numpy(sample['onehot_label']).long()}

            return {'T1': torch.from_numpy(T1), 'T1ce': torch.from_numpy(T1ce), 'T2w': torch.from_numpy(T2w),
                    'T2f': torch.from_numpy(T2f), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'T1': torch.from_numpy(T1), 'T1ce': torch.from_numpy(T1ce), 'T2w': torch.from_numpy(T2w),
                    'T2f': torch.from_numpy(T2f), 'label': torch.from_numpy(sample['label']).long()}
            # return {'image': torch.from_numpy(image), 'label': torch.from_numpy(label).long()}
class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)