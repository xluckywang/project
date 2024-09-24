# *_*coding: utf-8 *_*
# Author --wxq--

import os
import random
import shutil


def dataset_split(origion_path, save_train_dir, save_test_dir, train_rate):
    # 定义复制文件函数
    def copyFile(fileDir, class_name):
        image_list = os.listdir(fileDir)
        image_number = len(image_list)

        train_number = int(image_number * train_rate)
        train_sample = random.sample(image_list, train_number)  # 从image_list中随机获取0.8比例的图像.
        test_sample = list(set(image_list) - set(train_sample))
        sample = [train_sample, test_sample]

        # 复制图像到目标文件夹
        for k in range(len(save_dir)):
            class_folder=os.path.join(save_dir[k],class_name)
            if os.path.isdir(class_folder):
                for name in sample[k]:

                    shutil.copy(os.path.join(fileDir, name), os.path.join(class_folder , name))
            else:

                os.makedirs(class_folder,exist_ok=True)
                for name in sample[k]:
                    shutil.copy(os.path.join(fileDir, name), os.path.join(class_folder, name))

    # 原始数据集路径
    save_dir = [save_train_dir, save_test_dir]

    # 数据集类别及数量
    file_list = os.listdir(origion_path)
    num_classes = len(file_list)

    for i in range(num_classes):
        class_name = file_list[i]
        image_Dir = os.path.join(origion_path, class_name)
        copyFile(image_Dir, class_name)
        print('%s划分完毕！' % class_name)
