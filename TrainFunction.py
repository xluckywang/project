# *_*coding: utf-8 *_*
# author
#https://blog.csdn.net/limingkaoyan/article/details/106193549?spm=1001.2014.3001.5502

import os
import shutil
import time
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.models as models
import matplotlib
matplotlib.use('TkAgg')  # 使用TkAgg后端
import matplotlib.pyplot as plt

import ConfigSetting
from DataLoader import data_train, data_test
from DataSplit import dataset_split

os.environ['CUDA_VISION_DEVICES'] = '0'


# 训练函数
def model_train(model, train_data_load, optimizer, loss_func, epoch, log_interval):
    model.train()

    correct = 0
    train_loss = 0
    total = len(train_data_load.dataset)

    for i, (img, label) in enumerate(train_data_load, 0):
        begin = time.time()
        img, label = img.to(device), label.to(device)

        optimizer.zero_grad()

        outputs = model(img)
        loss = loss_func(outputs, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == label).sum()

        if (i + 1) % log_interval == 0:
            traind_total = (i + 1) * len(label)
            acc = 100. * correct / traind_total
            end = time.time()

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}\t lr: {}\t Train_Acc: {:.6f}\t Speed: {}'.format(
                epoch,
                i * len(img),
                total,
                100. * i / len(train_data_load),
                loss.data.item(),
                optimizer.param_groups[0]['lr'],
                acc,
                end - begin))

            global_train_acc.append(acc)


def model_test(model, test_data_load, epoch, kk,loss_func):
    model.eval()

    correct = 0
    test_loss=0
    test_losses=[]
    total = len(test_data_load.dataset)

    for i, (img, label) in enumerate(test_data_load):
        img, label = img.to(device), label.to(device)

        outputs = model(img)
        loss=loss_func(outputs,label)
        test_loss+=loss.item()
        _, pre = torch.max(outputs.data, 1)
        correct += (pre == label).sum()
    test_losses.append(test_loss/total)

    acc = correct.item() * 100. / (len(test_data_load.dataset))
    # 记录最佳分类精度
    global best_acc
    if acc > best_acc:
        best_acc = acc

    print('\nTest Set: Accuracy: {}/{}, ({:.6f}%)\nBest_Acc: {}\n'.format(correct, total, acc, best_acc))
    global_test_acc.append(acc)
    if best_acc > ConfigSetting.stop_accuracy:
        torch.save(model.state_dict(), str(kk + 1) + '_ResNet34_BestScore_' + str(best_acc) + '.pth')


def show_acc_curv(ratio, kk):
    # 训练准确率曲线的x、y
    train_x = list(range(len(global_train_acc)))
    train_y = global_train_acc
    # 将损失曲线从GPU转移到CPU
    train_y = [loss.cpu() for loss in train_y]

    # 测试准确率曲线的x、y
    # 每ratio个训练准确率对应一个测试准确率
    test_x = train_x[ratio - 1::ratio]
    test_y = global_test_acc

    plt.title('M ResNet34 ACC')
    plt.plot(train_x, train_y, color='green', label='training accuracy')
    plt.plot(test_x, test_y, color='red', label='testing accuracy')
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('accs')
    plt.savefig(ConfigSetting.img_save_path + 'acc_curv_' + str(kk + 1) + '.jpg')
    plt.show()


def adjust_learning_rate(optimizer, epoch):
    if epoch % ConfigSetting.adjust_lr_epoch == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1


if __name__ == '__main__':
    # 按设定的划分比例,得到随机数据集,分别进行10次训练,然后计算其均值和方差.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for k in range(10):

        best_acc = 0
        global_train_acc = []
        global_test_acc = []

        # 如果k=0, 则表示对第一次随机划分进行训练.
        dataset_split(ConfigSetting.origion_path, ConfigSetting.save_train_dir, ConfigSetting.save_test_dir, ConfigSetting.train_rate)

        # 加载数据集
        trainset, train_loader = data_train(ConfigSetting.save_train_dir)
        testset, test_loader = data_test(ConfigSetting.save_test_dir)
        print('\n数据加载完毕，开始训练\n')
        cudnn.benchmark = True

        # 加载模型
        """
        # VGG16
        model = models.vgg16(pretrained=True)
        model.classifier[-1].out_features = ConfigSetting.classes_num
        model = model.cuda()
        """
        # ResNet
        model = models.resnet18(pretrained=True)
        fc_features = model.fc.in_features
        model.fc = torch.nn.Linear(fc_features, ConfigSetting.classes_num)
        # model = model.cuda()
        model=model.to(device)

        # 优化器与损失
        optimizer = optim.SGD(model.parameters(), lr=ConfigSetting.init_lr, momentum=ConfigSetting.momentum)
        # loss_func = nn.CrossEntropyLoss().cuda()
        loss_func = nn.CrossEntropyLoss().to(device)
        start_time = time.time()

        # 训练
        for epoch in range(1, ConfigSetting.epochs + 1):
            print('----------------------第%s轮----------------------------' % epoch)
            model_train(model, train_loader, optimizer, loss_func, epoch, ConfigSetting.log_interval)
            model_test(model, test_loader, epoch, k,loss_func)
            adjust_learning_rate(optimizer, epoch)

        end_time = time.time()
        print('Train Speed Time:', end_time - start_time)

        # 显示训练和测试曲线
        ratio = len(trainset) / ConfigSetting.train_batch_size / ConfigSetting.log_interval
        ratio = int(ratio)
        show_acc_curv(ratio, k)

        # 保存模型
        torch.save(model.state_dict(), 'resnet_fs_' + str(k+1) + '.pth')

        ## 清除数据文件夹中的文件
        RS_path = ConfigSetting.split_dataset_save_dir
        file_list = os.listdir(RS_path)
        for m in range(len(file_list)):
            shutil.rmtree(RS_path + file_list[m])

        print('第{}次训练结束, 最佳分类精度为:{}'.format(k + 1, best_acc))
        print('--------------------------------------------------\n')

    print('全部训练完毕！')


