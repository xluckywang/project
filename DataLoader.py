# -*- coding: utf-8 -*-
# author ---wxq---

import torch
import torchvision
from PIL import Image

import ConfigSetting


def data_train(train_dataset):
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((ConfigSetting.resize, ConfigSetting.resize), Image.BILINEAR),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(ConfigSetting.crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))
    ])
    train_data = torchvision.datasets.ImageFolder(root=train_dataset,
                                                  transform=train_transforms)
    CLASS = train_data.class_to_idx
    print('训练数据label与文件名的关系为:', CLASS)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=ConfigSetting.train_batch_size,
                                               shuffle=True)
    return train_data, train_loader


def data_test(test_dataset):
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((ConfigSetting.resize, ConfigSetting.resize), Image.BILINEAR),
        torchvision.transforms.CenterCrop(ConfigSetting.crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))
    ])
    test_data = torchvision.datasets.ImageFolder(root=test_dataset,
                                                 transform=test_transforms)
    CLASS = test_data.class_to_idx
    print('验证数据label与文件名的关系为:', CLASS)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=1,
                                              shuffle=False)
    return test_data, test_loader


if __name__ == '__main__':
    data_train()
    data_test()
