# coding:utf-8
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from Index_calculation import *
import random
# X_train = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
G = testclass()

# 微分熵 + 归一化
# DE_Normalization = np.load("D:/Work_2/SEED-VIG/Raw_Data/dataset_DE/DE_Normalization.npy")
# DE_Normalization = np.load("D:/Work_2/SEED-VIG/Raw_Data/dataset_DE/data_DE.npy")
DE_Normalization = np.load("D:/Work_2/MyDataset/mix_data.npy")
DE_Normalization = DE_Normalization.reshape(3600,5,31,100)
print(DE_Normalization.shape)
# DE_Normalization = np.load("D:/pytorch/single-DE/singleDE-Normalization/dataDE-Nor22.npy")
# print(DE_Normalization.shape)
#
# # DE_Normalization = DE_Normalization.reshape(885, 17, 5, 368)
# DE_Normalization = DE_Normalization.reshape(20355, 5, 17, 16)
# print(DE_Normalization.shape)
# # print(DE_Normalization.shape)
DE_Normalization = torch.FloatTensor(DE_Normalization)
# # DE_Normalization = DE_Normalization.unsqueeze(1)
#
#
# 标签
label = np.load('D:/Work_2/MyDataset/mix_label.npy')
print(label.shape)
# label = np.load('D:/pytorch/single-DE/single-label/EEGlabel22.npy')
# print(label.shape)
label = torch.FloatTensor(label)

#SEED-VIG混合实验
# X_train = np.load("D:/Work_2/DataTrain/X_train.npy")
# X_train = torch.FloatTensor(X_train)
# print(X_train.shape)
# X_test = np.load("D:/Work_2/DataTest/X_test.npy")
# X_test = torch.FloatTensor(X_test)
# print(X_test.shape)
# Y_train = np.load("D:/Work_2/LabelTrain/Y_train.npy")
# Y_train = torch.FloatTensor(Y_train)
# print(Y_train.shape)
# Y_test = np.load("D:/Work_2/LabelTest/Y_test.npy")
# Y_test = torch.FloatTensor(Y_test)
# print(Y_test.shape)

#SEED-VIG跨被试实验
# X_train = np.load("D:/Work_2/SEED-VIG/Raw_Data/DE_mix/DE_Ex22.npy")
# X_train = X_train.reshape(19470,5,17,16)
# X_train = torch.FloatTensor(X_train)
# print(X_train.shape)
# X_test = np.load("D:/Work_2/SEED-VIG/Raw_Data/dataset_DE/DE_Normalization_22.npy")
# X_test = X_test.reshape(885,5,17,16)
# X_test = torch.FloatTensor(X_test)
# print(X_test.shape)
# Y_train = np.load("D:/Work_2/SEED-VIG/perclos_labels/label_Ex22.npy")
# Y_train = Y_train.reshape(19470,1)
# Y_train = torch.FloatTensor(Y_train)
# print(Y_train.shape)
# Y_test = np.load("D:/Work_2/SEED-VIG/perclos_labels/labels22.npy")
# Y_test = torch.FloatTensor(Y_test)
# print(Y_test.shape)

#Mydataset跨被试实验
# X_train = np.load("D:/Work_2/MyDataset/cross_subject_data/Exceptionyw_DENorm.npy")
# X_train = X_train.reshape(3240,5,31,100)
# X_train = torch.FloatTensor(X_train)
# print(X_train.shape)
# X_test = np.load("D:/Work_2/MyDataset/10名被试第一小时完整数据/yw_DENorm.npy")
# X_test = X_test.reshape(360,5,31,100)
# X_test = torch.FloatTensor(X_test)
# print(X_test.shape)
# Y_train = np.load("D:/Work_2/MyDataset/cross_subject_label/Exceptionyw.npy")
# Y_train = torch.FloatTensor(Y_train)
# print(Y_train.shape)
# Y_test = np.load("D:/Work_2/MyDataset/10名被试第一小时完整标签/yw1_1.npy")
# Y_test = torch.FloatTensor(Y_test)
# print(Y_test.shape)

#Mydataset混合实验
# X_train = np.load("D:/Work_2/MyDataset/混合实验数据训练集/mix_train.npy")
# X_train = torch.FloatTensor(X_train)
# print(X_train.shape)
# X_test = np.load("D:/Work_2/MyDataset/混合实验数据测试集/mix_test.npy")
# X_test = torch.FloatTensor(X_test)
# print(X_test.shape)
# Y_train = np.load("D:/Work_2/MyDataset/混合实验标签训练集/mix_train.npy")
# Y_train = torch.FloatTensor(Y_train)
# print(Y_train.shape)
# Y_test = np.load("D:/Work_2/MyDataset/混合实验标签测试集/mix_test.npy")
# Y_test = torch.FloatTensor(Y_test)
# print(Y_test.shape)

# 用于将矩阵随机划分为训练子集和测试子集，并返回划分好的训练集测试集样本和训练集测试集标签
# train_X, test_X为文本数据， train_Y, test_Y为标签数据
X_train, X_test, Y_train, Y_test = train_test_split(DE_Normalization, label, test_size=0.1, random_state=1600)

print("训练集测试集已划分完成............")
batch_size = 128
trainData =TensorDataset(X_train, Y_train)
testData = TensorDataset(X_test, Y_test)
train_dataloader = DataLoader(trainData, batch_size=batch_size, shuffle=True,drop_last=True)
test_dataloader = DataLoader(testData, batch_size=batch_size, shuffle=True,drop_last=True)
print("dataloader已完成装载............")



