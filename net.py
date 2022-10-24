# coding:utf-8
import warnings
from Data import *

import torch
from torch import nn
import matplotlib.pyplot as plt
from Index_calculation import *
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

from pylab import *
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")

batch_size = 128
learning_rate = 0.001
epochs = 200
min_acc = 0.7
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EEG_ConvNeXt(nn.Module):

    def __init__(self, in_chans=5, num_classes=1000
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() 
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=1, stride=4),
            nn.BatchNorm2d(dims[0]),
            LayerNorm(dims[0], eps=1e-6)
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=1, stride=2),
                nn.BatchNorm2d(dims[i+1])
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(3):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)
        self.bilstm = nn.LSTM(input_size=5*31*100, hidden_size=20, num_layers=2, bidirectional=True)
        self.fc1 = nn.Linear(1040,100)
        self.fc2 = nn.Linear(100,1)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(3):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        out = self.forward_features(x)
        out = self.head(out)

        y = x.reshape(-1, 1, 5*31*100)
        y, (_a, _b) = self.bilstm(y)
        y = F.gelu(y)
        s, b, h = y.shape
        y = y.view(s * b, h)
        y = y.reshape(-1, 40)

        z = torch.cat([out, y], dim=1)

        z = self.fc1(z)
        z = F.dropout(z,0.5)
        z = self.fc2(z)

        return z




myModel = EEG_ConvNeXt().to(device)
loss_func = nn.MSELoss().to(device)
opt = torch.optim.Adam(myModel.parameters(), lr=learning_rate)

G = testclass()
train_len = G.len(X_train.shape[0], batch_size)
test_len = G.len(X_test.shape[0], batch_size)

train_loss_plt = []
train_acc_plt = []
test_loss_plt = []
test_acc_plt = []
Train_Loss_list = []
Train_Accuracy_list = []
Test_Loss_list = []
Test_Accuracy_list = []

for i in range(epochs):
    total_train_step = 0
    total_test_step = 0

    total_train_loss = 0
    total_train_acc = 0

    for data in train_dataloader:
        x, y = data
        x = x.to(device)
        y = y.to(device)
        outputs = myModel(x)
        train_loss = loss_func(y, outputs)

        opt.zero_grad()
        train_loss.backward()
        opt.step()

        train_label = G.train_lable2(outputs)
        label = G.train_lable2(y)
        train_acc = G.acc(train_label, label)

        train_loss_plt.append(train_loss)
        total_train_loss = total_train_loss + train_loss.item()
        total_train_step = total_train_step + 1

        train_acc_plt.append(train_acc)
        total_train_acc += train_acc

    Train_Loss_list.append(total_train_loss / (len(train_dataloader)))
    Train_Accuracy_list.append(total_train_acc / train_len)

    total_test_loss = 0
    total_test_acc = 0
    matrix = [0, 0, 0, 0]
    with torch.no_grad():
        pred_output_list = []

        for data in test_dataloader:
            testx, testy = data
            testx = testx.to(device)
            testy = testy.to(device)
            outputs = myModel(testx)
            test_loss = loss_func(testy, outputs)

            test_label = G.train_lable2(outputs)
            label = G.train_lable2(testy)
            test_acc = G.acc(test_label, label)
            TP_TN_FP_FN = G.Compute_TP_TN_FP_FN(test_label, label, matrix)

            test_loss_plt.append(test_loss)
            total_test_loss = total_test_loss + test_loss.item()
            total_test_step = total_test_step + 1

            test_acc_plt.append(test_acc)
            total_test_acc += test_acc




    Test_Loss_list.append(total_test_loss / (len(test_dataloader)))
    Test_Accuracy_list.append(total_test_acc / test_len)

    if(total_test_acc / test_len) > min_acc:
        min_acc = total_test_acc / test_len
        res_TP_TN_FP_FN = TP_TN_FP_FN
        torch.save(myModel.state_dict(), 'D:/Work_2/MyDataset/mix.pth')


    print("Epoch: {}/{} ".format(i + 1, epochs),
          "Training Loss: {:.4f} ".format(total_train_loss / len(train_dataloader)),
          "Training Accuracy: {:.4f} ".format(total_train_acc / train_len),
          "Test Loss: {:.4f} ".format(total_test_loss / len(test_dataloader)),
          "Test Accuracy: {:.4f}".format(total_test_acc / test_len)
          )


print(min_acc)
print("TP: {}".format(res_TP_TN_FP_FN[0]))
print("TN: {}".format(res_TP_TN_FP_FN[1]))
print("FP: {}".format(res_TP_TN_FP_FN[2]))
print("FN: {}".format(res_TP_TN_FP_FN[3]))

train_x1 = range(0, 200)
train_x2 = range(0, 200)
train_y1 = Train_Accuracy_list
train_y2 = Train_Loss_list
plt.subplot(2, 1, 1)
plt.plot(train_x1, train_y1, 'o-')
plt.title('Train accuracy vs. epoches')
plt.ylabel('Train accuracy')
plt.subplot(2, 1, 2)
plt.plot(train_x2, train_y2, '.-')
plt.xlabel('Train loss vs. epoches')
plt.ylabel('Train loss')
plt.show()

test_x1 = range(0, 200)
test_x2 = range(0, 200)
test_y1 = Test_Accuracy_list
test_y2 = Test_Loss_list
plt.subplot(2, 1, 1)
plt.plot(test_x1, test_y1, 'o-')
plt.title('Test accuracy vs. epoches')
plt.ylabel('Test accuracy')
plt.subplot(2, 1, 2)
plt.plot(test_x2, test_y2, '.-')
plt.xlabel('Test loss vs. epoches')
plt.ylabel('Test loss')
plt.show()







