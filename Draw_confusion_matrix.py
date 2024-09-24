# _*_ coding: UTF-8 _*_
# Author

import torch
import numpy as np
import torchvision.models as models
from sklearn.metrics import confusion_matrix

import ConfigSetting
from DataLoader import data_test
from confusion_matrix import plot_confusion_matrix
import os

# 模型权重和类别标签
# weight_path = 'F:\\my_project\\scene_classification\\paper_v1\\1_ResNet34_BestScore_96.28571428571429.pth'
weight_path=os.path.join(ConfigSetting.weight_path,'1_ResNet34_BestScore_100.0.pth')
classes = ['bubble', 'dust', 'fouling', 'pinhole', 'sagging', 'scratch', 'shrink']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def LoadNet(weight_path):
    net = models.resnet18(pretrained=False)
    fc_features = net.fc.in_features
    net.fc = torch.nn.Linear(fc_features, ConfigSetting.classes_num)
    net.load_state_dict(torch.load(weight_path))
    net.eval()
    net.to(device)
    return net


# 导入测试数据集
# testset, test_loader = data_test(ConfigSetting.save_test_dir)
# val_dir='C:\\Users\\Administrator\\Desktop\\AI_learning\\00#pytorch\\data\\test'
result_gan='C:\\Users\\Administrator\\Desktop\\AI_learning\\00#pytorch\\Results_GANs'
testset, test_loader = data_test(result_gan)
print('\n测试数据加载完毕\n')

true_label = []
pred_label = []

# 加载模型
model = LoadNet(weight_path)

for batch_idx, (image, label) in enumerate(test_loader):
    image, label = image.to(device), label.to(device)
    output = model(image)
    pred = output.data.max(1, keepdim=True)[1]
    prediction = pred.squeeze(1)
    prediction = prediction.cpu().numpy()
    label = label.cpu().numpy()
    print('输入图像的真实标签为:{}, 预测标签为:{}'.format(label[0] + 1, prediction[0] + 1))
    true_label.append(label[0] + 1)
    pred_label.append(prediction[0] + 1)

# 计算混淆矩阵并绘图
cm = confusion_matrix(true_label, pred_label)
plot_confusion_matrix(classes, cm, 'confusion_matrix_96.jpg', title='confusion matrix')
