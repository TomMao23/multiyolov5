###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
# 带标准数据加载增广的语义分割Dataset, Dataset类代码原作者张航, 详见其开发的库 PyTorch-Encoding
# 稍加修改即可加载BDD100k分割数据
###########################################################################

import os
from tqdm import tqdm, trange
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
import torch.utils.data as data
from torchvision import transforms


# 基础语义分割类, 各数据集可以继承此类实现
class BaseDataset(data.Dataset):
    def __init__(self, root, split, mode=None, transform=None,
                 target_transform=None, base_size=520, crop_size=480):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size
        if self.mode == 'train':
            print('BaseDataset: base_size {}, crop_size {}'. \
                format(base_size, crop_size))

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

    def _val_sync_transform(self, img, mask):  # 训练中验证数据处理: 把图短边resize成crop_size, 长边保持比例, 再crop一块用于验证
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
        return img, self._mask_transform(mask)

    def _sync_transform(self, img, mask):  # 训练数据增广
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)  从base_size一半到两倍间随机取数, 图resize长边为此数, 短边保持比例
        w, h = img.size
        long_size = random.randint(int(self.base_size*0.66), int(self.base_size*1.66))
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
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)  # 为什么mask可以填0:类别0不是训练类别,后续会被填-1(bdd100k也适用)
        # random crop 随机按crop_size从resize和pad的图上crop一块用于训练
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1+crop_size, y1+crop_size))
        mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))
        # final transform
        return img, self._mask_transform(mask)

    def _mask_transform(self, mask):
        return torch.from_numpy(np.array(mask)).long()

# def test_batchify_fn(data):
#     error_msg = "batch must contain tensors, tuples or lists; found {}"
#     if isinstance(data[0], (str, torch.Tensor)):
#         return list(data)
#     elif isinstance(data[0], (tuple, list)):
#         data = zip(*data)
#         return [test_batchify_fn(i) for i in data]
#     raise TypeError((error_msg.format(type(batch[0]))))


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
        self._classes = np.array([0, 7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                                  23, 24, 25, 26, 27, 28, 31, 32, 33])
        self._key = np.array([-1, -1, -1, -1, -1, -1,
                              -1, -1,  0,  1, -1, -1,
                              2,   3,  4, -1, -1, -1,   # 35类, 训练类共19类, -1不是训练类
                              5,  -1,  6,  7,  8,  9,
                              10, 11, 12, 13, 14, 15,
                              -1, -1, 16, 17, 18])
        self._mapping = np.array(range(-1, len(self._key)-1)).astype('int32')

    def _class_to_index(self, mask):
        # assert the values
        values = np.unique(mask)
        for i in range(len(values)):
            assert(values[i] in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

    def _preprocess(self, mask_file):
        if os.path.exists(mask_file):
            masks = torch.load(mask_file)
            return masks
        masks = []
        print("Preprocessing mask, this will take a while." + \
            "But don't worry, it only run once for each split.")
        tbar = tqdm(self.mask_paths)
        for fname in tbar:
            tbar.set_description("Preprocessing masks {}".format(fname))
            mask = Image.fromarray(self._class_to_index(
                np.array(Image.open(fname))).astype('int8'))
            masks.append(mask)
        torch.save(masks, mask_file)
        return masks

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        #mask = self.masks[index]
        mask = Image.open(self.mask_paths[index])
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)  # 训练数据增广
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)  # 验证数据处理
        else:
            assert self.mode == 'testval'   # 训练时候验证用val(快, 省显存),测试验证集指标时用testval一般mIoU会更高且更接近真实水平
            mask = self._mask_transform(mask)  # 测试验证指标, 除转换标签格式外不做任何处理
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask

    def _mask_transform(self, mask):
        #target = np.array(mask).astype('int32') - 1
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


def get_city_pairs(folder, split='train'):
    def get_path_pairs(img_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for root, directories, files in os.walk(img_folder):
            for filename in files:
                if filename.endswith(".png"):
                    imgpath = os.path.join(root, filename)
                    foldername = os.path.basename(os.path.dirname(imgpath))
                    maskname = filename.replace('leftImg8bit','gtFine_labelIds')
                    maskpath = os.path.join(mask_folder, foldername, maskname)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
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


def get_citys_loader(root=os.path.expanduser('data/citys/'), split="train", mode="train",  # 获取训练和验证用的dataloader
                     base_size=2048, crop_size=768,
                     batch_size=32, workers=4, pin=True):
    if mode == "train":
        input_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.3, hue=0.1),
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
                               base_size=base_size, crop_size=crop_size)

    loader = data.DataLoader(dataset, batch_size=batch_size,
                             drop_last=True if mode == "train" else False, shuffle=True if mode == "train" else False,
                             num_workers=workers, pin_memory=pin)
    return loader


if __name__ == "__main__":
    trainloader, valloader = get_citys_loader(root='./data/citys/detdata', workers=4, pin=False, batch_size=16)
    from utils.loss import SegmentationLosses
    criteria = SegmentationLosses(se_loss=False,
                                  aux=False,
                                  nclass=19,
                                  se_weight=0.2,
                                  aux_weight=0.2)
    import time
    t1 = time.time()
    for i, data in enumerate(trainloader):
        print(f"batch: {i}")
    print(f"cost {(time.time()-t1)/(i+1)} per batch load")