# *_*coding: utf-8 *_*
# author

import os

########################### 数据划分参数 ############################
# origion_path = 'F:\\my_project\\scene_classification\\scene_classification_dataset\\UCMerced_LandUse\\'
# save_train_dir = 'F:\\my_project\\scene_classification\\paper_v1\\data\\RS_UC\\5_5\\train\\'
# save_test_dir = 'F:\\my_project\\scene_classification\\paper_v1\\data\\RS_UC\\5_5\\test\\'
origion_path=F'C:\\Users\\Administrator\\Desktop\\AI_learning\\00#pytorch\\data\\train'
save_train_dir=F'C:\\Users\\Administrator\\Desktop\\AI_learning\\00#pytorch\\Project\\data\\train'
save_test_dir=F'C:\\Users\\Administrator\\Desktop\\AI_learning\\00#pytorch\\Project\\data\\test'

train_rate = 0.8

split_dataset_save_dir = 'C:\\Users\\Administrator\\Desktop\\AI_learning\\00#pytorch\\Project\\data\\'
weight_path='C:\\Users\\Administrator\\Desktop\\AI_learning\\00#pytorch\\Project'
########################### 网络训练参数 ##################################
project_path = os.getcwd()

resize = 256
crop_size = 224

classes_num = 21

train_batch_size = 48
test_batch_size = 48
epochs = 10
init_lr = 0.001
momentum = 0.9
log_interval = 10
stop_accuracy = 90.00
adjust_lr_epoch = 60

img_save_path = project_path + '/fig/'