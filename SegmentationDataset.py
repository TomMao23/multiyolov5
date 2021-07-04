###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
# 带标准数据加载增广的语义分割Dataset, Dataset类代码原作者张航, 详见其开发的github仓库PyTorch-Encoding, 在此基础上魔改了一些包括不均匀的长边采样,色彩变换,pad0改成了pad255(配合bdd的格式)
# 稍加修改即可加载BDD100k分割数据, 此处写了Cityscapes+BDD100k混合训练，没加单独的BDD100k
###########################################################################

import os
from tqdm import tqdm, trange
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
import torch.utils.data as data
from torchvision import transforms
from utils.general import make_divisible
from scipy import stats
import math
from functools import lru_cache
import matplotlib.pyplot as plt
from random import choices


@lru_cache(128)  # 目前每次调用参数都是一样的, 用cache加速, 有random的地方不能用cache
def range_and_prob(base_size, low: float = 0.5,  high: float = 3.0, std: int = 25) -> list:
    low = math.ceil((base_size * low) / 32)
    high = math.ceil((base_size * high) / 32)
    mean = math.ceil(base_size / 32) - 4  # 峰值略偏
    x = np.array(list(range(low, high + 1)))
    p = stats.norm.pdf(x, mean, std)
    p = p / p.sum()  # 概率密度　choices权重不用归一化, 归一化用于debug和可视化调参std,以及用cum_weights优化
    cum_p = np.cumsum(p)  # 概率分布，累加
    # print("!!!!!!!!!!!!!!!!!!!!!!")
    return (x, cum_p)


# 用均值为basesize的正态分布模拟一个类似F分布图形的采样, 目的是专注于目标scale的同时见过少量大scale(通过apollo图天空同时不掉点)
def get_long_size(base_size:int, low: float = 0.5,  high: float = 3.0, std: int = 40) -> int:  
    x, cum_p = range_and_prob(base_size, low, high, std)
    # plt.plot(x, p)
    # plt.show()
    longsize = choices(population=x, cum_weights=cum_p, k=1)[0] * 32  # 用cum_weights O(logn)，　用weights O(n)
    # print(longsize)
    return longsize


# 基础语义分割类, 各数据集可以继承此类实现
class BaseDataset(data.Dataset):
    def __init__(self, root, split, mode=None, transform=None,
                 target_transform=None, base_size=520, crop_size=480, low=0.6, high=3.0, sample_std=25):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size
        self.low = low
        self.high = high
        self.sample_std = sample_std
        if self.mode == 'train':
            print('BaseDataset: base_size {}, crop_size {}'. \
                format(base_size, crop_size))
            print(f"Random scale low: {self.low}, high: {self.high}, sample_std: {self.sample_std}")

    def __getitem__(self, index):
        raise NotImplemented

    @property
    def num_class(self):
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        raise NotImplemented

    def make_pred(self, x):
        return x + self.pred_offset

    def _testval_img_transform(self, img):  # 新的训练后测验证集数据处理(仅支持同尺寸图): 图长边resize到base_size, 但标签是原图, 若非原图需要测试时手动把输出放大到原图 (原版仅处理标签, 原图输入)
        w, h = img.size
        outlong = self.base_size
        outlong = make_divisible(outlong, 32)  # 32是网络最大下采样倍数, 测试时自动使边为32倍数
        if w > h:
            ow = outlong
            oh = int(1.0 * h * ow / w)
            oh = make_divisible(oh, 32)
        else:
            oh = outlong
            ow = int(1.0 * w * oh / h)
            ow = make_divisible(ow, 32)
        img = img.resize((ow, oh), Image.BILINEAR)
        return img

    def _val_sync_transform(self, img, mask):  # 训练中验证数据处理(支持不同尺寸图，但是指标通常比testval略低一点点): 把图短边resize成crop_size, 长边保持比例, 再crop一块(crop_size,crop_size)用于验证(在citysbdd和custom中图不同时候使用)
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1+outsize, y1+outsize))
        mask = mask.crop((x1, y1, x1+outsize, y1+outsize))
        # final transform
        # return img, self._mask_transform(mask)
        return img, mask  # 这里改了, 在__getitem__里再调用self._mask_transform(mask)

    def _sync_transform(self, img, mask):  # 训练数据增广
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        w_crop_size, h_crop_size = self.crop_size
        # random scale (short edge)  从base_size一半到两倍间随机取数, 图resize长边为此数, 短边保持比例
        w, h = img.size
        long_size = get_long_size(base_size=self.base_size, low=self.low, high=self.high, std=self.sample_std)  # random.randint(int(self.base_size*0.5), int(self.base_size*2))
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop  边长比crop_size小就pad
        if ow < w_crop_size or oh < h_crop_size:  # crop_size:
            padh = h_crop_size - oh if oh < h_crop_size else 0
            padw = w_crop_size - ow if ow < w_crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)  # mask不填充0而是填255:类别0不是训练类别,后续会被填-1(但bdd100k数据格式是trainid,为了兼容填255)
        # random crop 随机按crop_size从resize和pad的图上crop一块用于训练
        w, h = img.size
        x1 = random.randint(0, w - w_crop_size)
        y1 = random.randint(0, h - h_crop_size)
        img = img.crop((x1, y1, x1+w_crop_size, y1+h_crop_size))
        mask = mask.crop((x1, y1, x1+w_crop_size, y1+h_crop_size))
        # final transform
        # return img, self._mask_transform(mask)
        return img, mask  # 这里改了, 在__getitem__里再调用self._mask_transform(mask)

    def _mask_transform(self, mask):
        return torch.from_numpy(np.array(mask)).long()


class CitySegmentation(BaseDataset):  # base_size 2048 crop_size 768
    NUM_CLASS = 19

    # mode训练时候验证用val, 测试验证集指标时候用testval一般会更高且更接近真实水平
    def __init__(self, root=os.path.expanduser('../data/citys/'), split='train',
                 mode=None, transform=None, target_transform=None, **kwargs):
        super(CitySegmentation, self).__init__(
            root, split, mode, transform, target_transform, **kwargs)
        # self.root = os.path.join(root, self.BASE_DIR)
        self.images, self.mask_paths = get_city_pairs(self.root, self.split)
        assert (len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of: \
                " + self.root + "\n")
        self._indices = np.array(range(-1, 19))
        self._classes = np.array([0, 7, 8, 11, 12, 13, 17, 19, 20, 21, 22,  # 这个不用管,用于测试集提交转标签的
                                  23, 24, 25, 26, 27, 28, 31, 32, 33])
        self._key = np.array([-1, -1, -1, -1, -1, -1,
                              -1, -1,  0,  1, -1, -1,   # Cityscapes标注是id(Bdd100k标注是trian_id不需要转换,仅需把255换成-1忽略)
                              2,   3,  4, -1, -1, -1,   # 35类, 训练类共19类, -1不是训练类
                              5,  -1,  6,  7,  8,  9,
                              10, 11, 12, 13, 14, 15,
                              -1, -1, 16, 17, 18])
        self._mapping = np.array(range(-1, len(self._key)-1)).astype('int32')

    def _class_to_index(self, mask):
        # assert the values
        mask[mask==255] = 0  # pad的255填充成0(id),　下面转trainid变成-1
        values = np.unique(mask)
        for i in range(len(values)):
            assert(values[i] in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        # mask = self.masks[index]
        mask = Image.open(self.mask_paths[index])
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)  # 训练数据增广
            mask = self._mask_transform(mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)  # 验证数据处理
            mask = self._mask_transform(mask)
        else:
            assert self.mode == 'testval'   # 训练时候验证用val(快, 省显存),测试验证集指标时用testval一般mIoU会更高且更接近真实水平
            # mask = self._mask_transform(mask)  # 测试验证指标, 除转换标签格式外不做任何处理
            img = self._testval_img_transform(img)
            mask = self._mask_transform(mask)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask

    def _mask_transform(self, mask):
        # target = np.array(mask).astype('int32') - 1
        target = self._class_to_index(np.array(mask).astype('int32'))
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.images)

    def make_pred(self, mask):
        values = np.unique(mask)
        for i in range(len(values)):
            assert(values[i] in self._indices)
        index = np.digitize(mask.ravel(), self._indices, right=True)
        return self._classes[index].reshape(mask.shape)


# 混合Cityscapes与BDD100k用这个, 把bdd当做Cityscapes的一个城市, 用jpg和png区分处理, 没写单独BDD的
class CityBddSegmentation(BaseDataset):  # base_size 2048 crop_size 768
    # mode训练时候验证用testval, 测试验证集指标时候也用testval, val倍废弃
    def __init__(self, root=os.path.expanduser('../data/citys/'), split='train',
                 mode=None, transform=None, target_transform=None, NUM_CLASS=19, **kwargs):
        super(CityBddSegmentation, self).__init__(
            root, split, mode, transform, target_transform, **kwargs)
        # self.root = os.path.join(root, self.BASE_DIR)
        self.images, self.mask_paths = get_city_pairs(self.root, self.split)
        assert (len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of: \
                " + self.root + "\n")
        self.NUM_CLASS = NUM_CLASS

        self._indices = np.array(range(-1, 19))
        self._classes = np.array([0, 7, 8, 11, 12, 13, 17, 19, 20, 21, 22,  #　这个不用管,用于测试集提交转标签的
                                  23, 24, 25, 26, 27, 28, 31, 32, 33])
        self._key = np.array([-1, -1, -1, -1, -1, -1,
                              -1, -1,  0,  1, -1, -1,   # Cityscapes标注是id(Bdd100k标注是trian_id不需要转换,仅需把255换成-1忽略)
                              2,   3,  4, -1, -1, -1,   # 35类, 训练类共19类, -1不是训练类
                              5,  -1,  6,  7,  8,  9,
                              10, 11, 12, 13, 14, 15,
                              -1, -1, 16, 17, 18])
        self._mapping = np.array(range(-1, len(self._key)-1)).astype('int32')

    def _class_to_index(self, mask):
        # assert the values
        mask[mask==255] = 0  # pad的255填充转成0(id),　下面转trainid变成-1
        values = np.unique(mask)
        for i in range(len(values)):
            assert(values[i] in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

    def __getitem__(self, index):
        imagepath = self.images[index]
        img = Image.open(imagepath).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        # mask = self.masks[index]
        mask = Image.open(self.mask_paths[index])
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)  # 训练数据增广
            if imagepath.endswith('png'):  # Cityscapes png　id转trian_id
                mask = self._mask_transform(mask)
            else:  # BDD100k jpg 只用把pad和原本忽略类的255替换成-1
                mask = torch.from_numpy(np.array(mask)).long()
                mask[mask==255] = -1
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)  # 验证数据处理
            if imagepath.endswith('png'):  # Cityscapes png　id转trian_id
                mask = self._mask_transform(mask)
            else:  # BDD100k jpg 只用把pad和原本忽略类的255替换成-1
                mask = torch.from_numpy(np.array(mask)).long()
                mask[mask==255] = -1
        else:
            assert self.mode == 'testval'   # 训练时候验证用val(快, 省显存),测试验证集指标时用testval一般mIoU会更高且更接近真实水平
            # mask = self._mask_transform(mask)  # 测试验证指标, 除转换标签格式外不做任何处理
            img = self._testval_img_transform(img)
            if imagepath.endswith('png'):  # Cityscapes png
                mask = self._mask_transform(mask)
            else:  # BDD100k jpg
                mask = torch.from_numpy(np.array(mask)).long()
                mask[mask==255] = -1

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask

    def _mask_transform(self, mask):
        # target = np.array(mask).astype('int32') - 1
        target = self._class_to_index(np.array(mask).astype('int32'))
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.images)

    def make_pred(self, mask):
        values = np.unique(mask)
        for i in range(len(values)):
            assert(values[i] in self._indices)
        index = np.digitize(mask.ravel(), self._indices, right=True)
        return self._classes[index].reshape(mask.shape)


class CustomSegmentation(BaseDataset):  # base_size 2048 crop_size 768
    # mode训练时候验证用testval, 测试验证集指标时候也用testval, val倍废弃
    def __init__(self, root=os.path.expanduser('../data/citys/'), split='train',
                 mode=None, transform=None, target_transform=None, **kwargs):
        super(CustomSegmentation, self).__init__(
            root, split, mode, transform, target_transform, **kwargs)
        # self.root = os.path.join(root, self.BASE_DIR)
        self.images, self.mask_paths = get_custom_pairs(self.root, self.split)
        assert (len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of: \
                " + self.root + "\n")

    def __getitem__(self, index):
        imagepath = self.images[index]
        img = Image.open(imagepath).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        # mask = self.masks[index]
        mask = Image.open(self.mask_paths[index])
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)  # 训练数据增广
            mask = torch.from_numpy(np.array(mask)).long()
            mask[mask==255] = -1
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)  # 验证数据处理
            mask = torch.from_numpy(np.array(mask)).long()
            mask[mask==255] = -1
        else:
            assert self.mode == 'testval'   # 训练时候验证用val(快, 省显存),测试验证集指标时用testval一般mIoU会更高且更接近真实水平
            # mask = self._mask_transform(mask)  # 测试验证指标, 除转换标签格式外不做任何处理
            img = self._testval_img_transform(img)
            mask = torch.from_numpy(np.array(mask)).long()
            mask[mask==255] = -1

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask

    def __len__(self):
        return len(self.images)


# 单独Cityscapes和混合Cityscapes与BDD100k都用这个, 把bdd当做Cityscapes的一个城市, 用jpg和png区分处理
def get_city_pairs(folder, split='train'):
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for root, directories, files in os.walk(img_folder):
            for filename in files:
                if filename.endswith(".png") or filename.endswith(".jpg"):
                    imgpath = os.path.join(root, filename)
                    foldername = os.path.basename(os.path.dirname(imgpath))
                    maskname = filename.replace('leftImg8bit', 'gtFine_labelIds')
                    if filename.endswith(".jpg"):  # BDD100k图是jpg，标签是png
                        maskname =maskname.replace('.jpg', '.png')
                    maskpath = os.path.join(mask_folder, foldername, maskname)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:  # 正常情况Cityscapes和BDD数据文件层面很干净不应该警告
                        print('cannot find the mask or image:', imgpath, maskpath)
        print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths

    if split == 'train' or split == 'val' or split == 'test':
        img_folder = os.path.join(folder, 'leftImg8bit/' + split)
        mask_folder = os.path.join(folder, 'gtFine/'+ split)
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        return img_paths, mask_paths
    else:
        assert split == 'trainval'
        print('trainval set')
        train_img_folder = os.path.join(folder, 'leftImg8bit/train')
        train_mask_folder = os.path.join(folder, 'gtFine/train')
        val_img_folder = os.path.join(folder, 'leftImg8bit/val')
        val_mask_folder = os.path.join(folder, 'gtFine/val')
        train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
        val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
        img_paths = train_img_paths + val_img_paths
        mask_paths = train_mask_paths + val_mask_paths
    return img_paths, mask_paths


def get_custom_pairs(folder, split='train'):
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for root, directories, files in os.walk(img_folder):
            for filename in files:
                if filename.endswith(".png") or filename.endswith(".jpg"):
                    imgpath = os.path.join(root, filename)
                    # foldername = os.path.basename(os.path.dirname(imgpath)) #customdata不用像cityscapes一样包装一个城市名字了
                    maskname = filename.replace('segimages', 'seglabels')
                    if filename.endswith(".jpg"):  # 图可以是jpg，但是标签必须是png
                        maskname =maskname.replace('.jpg', '.png')
                    # maskpath = os.path.join(mask_folder, foldername, maskname)
                    maskpath = os.path.join(mask_folder, maskname)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:  # 正常情况Cityscapes和BDD数据文件层面很干净不应该警告
                        print('cannot find the mask or image:', imgpath, maskpath)
        print('Found {} images in the folder {}'.format(len(img_paths), img_folder))
        return img_paths, mask_paths

    if split == 'train' or split == 'val' or split == 'test':
        img_folder = os.path.join(folder, 'segimages/' + split)
        mask_folder = os.path.join(folder, 'seglabels/'+ split)
        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        return img_paths, mask_paths
    else:
        assert split == 'trainval'
        print('trainval set')
        train_img_folder = os.path.join(folder, 'leftImg8bit/train')
        train_mask_folder = os.path.join(folder, 'gtFine/train')
        val_img_folder = os.path.join(folder, 'leftImg8bit/val')
        val_mask_folder = os.path.join(folder, 'gtFine/val')
        train_img_paths, train_mask_paths = get_path_pairs(train_img_folder, train_mask_folder)
        val_img_paths, val_mask_paths = get_path_pairs(val_img_folder, val_mask_folder)
        img_paths = train_img_paths + val_img_paths
        mask_paths = train_mask_paths + val_mask_paths
    return img_paths, mask_paths


def get_citys_loader(root=os.path.expanduser('data/citys/'), split="train", mode="train",  # 获取训练和验证用的dataloader
                     base_size=1024, crop_size=(1024, 512),
                     batch_size=32, workers=4, pin=True):
    if mode == "train":
        input_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.45, contrast=0.45,
                                   saturation=0.45, hue=0.15),
            transforms.ToTensor(),
            # transforms.Normalize([.485, .456, .406], [.229, .224, .225])  # 为了配合检测预处理保持一致, 分割不做norm
        ])
    else:
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([.485, .456, .406], [.229, .224, .225])  # 为了配合检测预处理保持一致, 分割不做norm
        ])
    dataset = CitySegmentation(root=root, split=split, mode=mode,
                               transform=input_transform,
                               base_size=base_size, crop_size=crop_size, low=0.65, high=3, sample_std=25)

    loader = data.DataLoader(dataset, batch_size=batch_size,
                             drop_last=  False, shuffle=True if mode == "train" else False,
                             num_workers=workers, pin_memory=pin)
    return loader


def get_citysbdd_loader(root=os.path.expanduser('data/citys/'), split="train", mode="train",  # 获取训练和验证用的dataloader
                     base_size=1024, crop_size=(1024, 512),
                     batch_size=32, workers=4, pin=True):
    if mode == "train":
        input_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                   saturation=0.4, hue=0.05),
            transforms.ToTensor(),
            # transforms.Normalize([.485, .456, .406], [.229, .224, .225])  # 为了配合检测预处理保持一致, 分割不做norm
        ])
    else:
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([.485, .456, .406], [.229, .224, .225])  # 为了配合检测预处理保持一致, 分割不做norm
        ])
    dataset = CityBddSegmentation(root=root, split=split, mode=mode,
                               transform=input_transform,
                               base_size=base_size, crop_size=crop_size, low=0.65, high=2, sample_std=40)

    loader = data.DataLoader(dataset, batch_size=batch_size,
                             drop_last=True if mode == "train" else False, shuffle=True if mode == "train" else False,
                             num_workers=workers, pin_memory=pin)
    return loader


# 默认custom_loader　jitter和crop采用更保守的方案
def get_custom_loader(root=os.path.expanduser('data/citys/'), split="train", mode="train",  # 获取训练和验证用的dataloader
                     base_size=1024,  # crop_size=(1024, 1024), 注意 custom的corpsize=basesize
                     batch_size=32, workers=4, pin=True):
    if mode == "train":
        input_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                   saturation=0.4, hue=0),
            transforms.ToTensor(),
            # transforms.Normalize([.485, .456, .406], [.229, .224, .225])  # 为了配合检测预处理保持一致, 分割不做norm
        ])
    else:
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([.485, .456, .406], [.229, .224, .225])  # 为了配合检测预处理保持一致, 分割不做norm
        ])
    dataset = CustomSegmentation(root=root, split=split, mode=mode,
                               transform=input_transform,
                               base_size=base_size, crop_size=(base_size, base_size), low=0.75, high=1.5, sample_std=35)

    loader = data.DataLoader(dataset, batch_size=batch_size,
                             drop_last=True if mode == "train" else False, shuffle=True if mode == "train" else False,
                             num_workers=workers, pin_memory=pin)
    return loader


if __name__ == "__main__":
    t = transforms.Compose([  # 用于打断点时候测试色彩和大小裁剪变换是否合理
        transforms.ColorJitter(brightness=0.45, contrast=0.45,
                               saturation=0.45, hue=0.1)])
    # trainloader = get_citys_loader(root='./data/citys/', split="val", mode="train", base_size=1024, crop_size=(832, 416), workers=0, pin=True, batch_size=4)
    trainloader = get_custom_loader(root='./data/customdata/', split="train", mode="train", base_size=832, workers=0, pin=True, batch_size=4)

    import time
    t1 = time.time()
    for i, data in enumerate(trainloader):
        print(f"batch: {i}")
    print(f"cost {(time.time()-t1)/(i+1)} per batch load")
    pass

    pass