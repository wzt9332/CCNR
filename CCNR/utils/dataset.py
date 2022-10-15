import os
from torch.utils.data.dataset import Dataset
from PIL import Image
from PIL import ImageFilter
import random
import torch
import torch.utils.data.sampler as sampler
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f
import numpy as np

def get_pascal_idx(root, train=True):
    root = os.path.expanduser(root)
    if train:
        file_name = root + '/train.txt'
    else:
        file_name = root + '/val.txt'
    with open(file_name) as f:
        idx_list = f.read().splitlines()
    if train:
        labeled_idx = idx_list
        return labeled_idx
    else:
        return idx_list

def transform(image, label, logits=None, crop_size=(512, 512), scale_size=(0.8, 1.0), augmentation=True):
    # Random rescale image
    raw_w, raw_h = image.size
    scale_ratio = random.uniform(scale_size[0], scale_size[1])

    resized_size = (int(raw_h * scale_ratio), int(raw_w * scale_ratio))
    image = transforms_f.resize(image, resized_size, Image.BILINEAR)
    label = transforms_f.resize(label, resized_size, Image.NEAREST)
    if logits is not None:
        logits = transforms_f.resize(logits, resized_size, Image.NEAREST)

    # Add padding if rescaled image size is less than crop size
    if crop_size == -1:  # use original im size without crop or padding
        crop_size = (raw_h, raw_w)

    if crop_size[0] > resized_size[0] or crop_size[1] > resized_size[1]:
        right_pad, bottom_pad = max(crop_size[1] - resized_size[1], 0), max(crop_size[0] - resized_size[0], 0)
        image = transforms_f.pad(image, padding=(0, 0, right_pad, bottom_pad), padding_mode='reflect')
        label = transforms_f.pad(label, padding=(0, 0, right_pad, bottom_pad), fill=255, padding_mode='constant')
        if logits is not None:
            logits = transforms_f.pad(logits, padding=(0, 0, right_pad, bottom_pad), fill=0, padding_mode='constant')

    # Random Cropping
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
    image = transforms_f.crop(image, i, j, h, w)
    label = transforms_f.crop(label, i, j, h, w)
    if logits is not None:
        logits = transforms_f.crop(logits, i, j, h, w)

    if augmentation:
        # Random color jitter
        if torch.rand(1) > 0.2:
            color_transform = transforms.ColorJitter((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25)) # For PyTorch 1.9/TorchVision 0.10 users
            # color_transform = transforms.ColorJitter.get_params((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))
            image = color_transform(image)

        # Random Gaussian filter
        if torch.rand(1) > 0.5:
            sigma = random.uniform(0.15, 1.15)
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))

        # Random horizontal flipping
        if torch.rand(1) > 0.5:
            image = transforms_f.hflip(image)
            label = transforms_f.hflip(label)
            if logits is not None:
                logits = transforms_f.hflip(logits)

    # Transform to tensor
    image = transforms_f.to_tensor(image)
    label = (transforms_f.to_tensor(label) * 255).long()
    label[label == 255] = -1  # invalid pixels are re-mapped to index -1
    if logits is not None:
        logits = transforms_f.to_tensor(logits)


    if logits is not None:
        return image, label, logits
    else:
        return image, label



class BuildDataset(Dataset):
    def __init__(self, root, idx_list, crop_size=(512, 512), scale_size=(0.5, 2.0),
                 augmentation=True, train=True, apply_partial=None, partial_seed=None):
        self.root = os.path.expanduser(root)
        self.train = train
        self.crop_size = crop_size
        self.augmentation = augmentation
        self.idx_list = idx_list
        self.scale_size = scale_size
        self.apply_partial = apply_partial
        self.partial_seed = partial_seed

    def __getitem__(self, index):
        image_root = Image.open(self.root + '/JPEGImages/{}.tiff'.format(self.idx_list[index]))
        # ---------图像3通道和1通道的转换
        image_root = image_root.convert(mode='RGB')
        # image_root = image_root.convert(mode='L')
        if self.apply_partial is None:
            label_root = Image.open(self.root + '/SegmentationClassAug/{}.png'.format(self.idx_list[index]))
        else:
            label_root = Image.open(self.root + '/SegmentationClassAug_{}_{}/{}.png'.format(self.apply_partial,  self.partial_seed, self.idx_list[index],))

        image, label = transform(image_root, label_root, None, self.crop_size, self.scale_size, self.augmentation)
        label = label.squeeze(0)
        return image, label

    def __len__(self):
        return len(self.idx_list)

def region_label(lable, crop_size = [16, 16]):
    crop_num0 = int(lable.shape[0] / crop_size[0])
    crop_num1 = int(lable.shape[1] / crop_size[1])
    lable_r = torch.zeros((crop_num0, crop_num1))
    for i in range(crop_num0):
        for j in range(crop_num1):
            lable_crop = lable[i * crop_size[0]: i * crop_size[0]+crop_size[0], j * crop_size[1]: j * crop_size[1]+crop_size[1]]
            sort = sorted([(torch.sum(lable_crop == w), w) for w in set(lable_crop.flatten())])
            lable_r[i,j] = sort[0][1]
    return lable_r


class BuildDataLoader:
    def __init__(self, dataset, batch_size, classes_num):
        self.data_path = dataset
        self.im_size = [256, 256]
        self.crop_size = [256, 256]
        self.num_segments = classes_num
        self.scale_size = (0.5, 1.5)
        self.batch_size = batch_size
        self.train_l_idx = get_pascal_idx(self.data_path, train=True)
        self.test_idx = get_pascal_idx(self.data_path, train=False)



    def build(self, supervised=False, partial=None, partial_seed=None):
        train_l_dataset = BuildDataset(self.data_path, self.train_l_idx,
                                       crop_size=self.crop_size, scale_size=self.scale_size,
                                       augmentation=True, train=True, apply_partial=partial, partial_seed=partial_seed)

        test_dataset    = BuildDataset(self.data_path, self.test_idx,
                                       crop_size=self.im_size, scale_size=(1.0, 1.0),
                                       augmentation=False, train=False)

        if supervised:  # no unlabelled dataset needed, double batch-size to match the same number of training samples
            self.batch_size = self.batch_size * 2

        num_samples = self.batch_size * 200  # for total 40k iterations with 200 epochs

        train_l_loader = torch.utils.data.DataLoader(
            train_l_dataset,
            batch_size=self.batch_size,
            sampler=sampler.RandomSampler(data_source=train_l_dataset,
                                          replacement=True,
                                          num_samples=num_samples),
            drop_last=True,
        )


        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=4,
            shuffle=False,
        )
        if supervised:
            return train_l_loader, test_loader
        else:
            return train_l_loader, test_loader




