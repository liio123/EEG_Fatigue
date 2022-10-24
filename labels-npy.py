import numpy as np
import scipy.io as sio


# LF1 = 'D:/Work_2/SEED-VIG/perclos_labels/1_20151124_noon_2.mat'
# LF2 = 'D:/Work_2/SEED-VIG/perclos_labels/2_20151106_noon.mat'
# LF3 = 'D:/Work_2/SEED-VIG/perclos_labels/3_20151024_noon.mat'
# LF4 = 'D:/Work_2/SEED-VIG/perclos_labels/4_20151105_noon.mat'
# LF5 = 'D:/Work_2/SEED-VIG/perclos_labels/4_20151107_noon.mat'
# LF6 = 'D:/Work_2/SEED-VIG/perclos_labels/5_20141108_noon.mat'
# LF7 = 'D:/Work_2/SEED-VIG/perclos_labels/5_20151012_night.mat'
# LF8 = 'D:/Work_2/SEED-VIG/perclos_labels/6_20151121_noon.mat'
# LF9 = 'D:/Work_2/SEED-VIG/perclos_labels/7_20151015_night.mat'
# LF10 = 'D:/Work_2/SEED-VIG/perclos_labels/8_20151022_noon.mat'
# LF11 = 'D:/Work_2/SEED-VIG/perclos_labels/9_20151017_night.mat'
# LF12 = 'D:/Work_2/SEED-VIG/perclos_labels/10_20151125_noon.mat'
# LF13 = 'D:/Work_2/SEED-VIG/perclos_labels/11_20151024_night.mat'
# LF14 = 'D:/Work_2/SEED-VIG/perclos_labels/12_20150928_noon.mat'
# LF15 = 'D:/Work_2/SEED-VIG/perclos_labels/13_20150929_noon.mat'
# LF16 = 'D:/Work_2/SEED-VIG/perclos_labels/14_20151014_night.mat'
# LF17 = 'D:/Work_2/SEED-VIG/perclos_labels/15_20151126_night.mat'
# LF18 = 'D:/Work_2/SEED-VIG/perclos_labels/16_20151128_night.mat'
# LF19 = 'D:/Work_2/SEED-VIG/perclos_labels/17_20150925_noon.mat'
# LF20 = 'D:/Work_2/SEED-VIG/perclos_labels/18_20150926_noon.mat'
# LF21 = 'D:/Work_2/SEED-VIG/perclos_labels/19_20151114_noon.mat'
# LF22 = 'D:/Work_2/SEED-VIG/perclos_labels/20_20151129_night.mat'
# LF23 = 'D:/Work_2/SEED-VIG/perclos_labels/21_20151016_noon.mat'
#
# labels1 = list(sio.loadmat(LF1).values())[3]
# # print(labels1.shape)
# labels2 = list(sio.loadmat(LF2).values())[3]
# labels3 = list(sio.loadmat(LF3).values())[3]
# labels4 = list(sio.loadmat(LF4).values())[3]
# labels5 = list(sio.loadmat(LF5).values())[3]
# labels6 = list(sio.loadmat(LF6).values())[3]
# labels7 = list(sio.loadmat(LF7).values())[3]
# labels8 = list(sio.loadmat(LF8).values())[3]
# labels9 = list(sio.loadmat(LF9).values())[3]
# labels10 = list(sio.loadmat(LF10).values())[3]
# labels11 = list(sio.loadmat(LF11).values())[3]
# labels12 = list(sio.loadmat(LF12).values())[3]
# labels13 = list(sio.loadmat(LF13).values())[3]
# labels14 = list(sio.loadmat(LF14).values())[3]
# labels15 = list(sio.loadmat(LF15).values())[3]
# labels16 = list(sio.loadmat(LF16).values())[3]
# labels17 = list(sio.loadmat(LF17).values())[3]
# labels18 = list(sio.loadmat(LF18).values())[3]
# labels19 = list(sio.loadmat(LF19).values())[3]
# labels20 = list(sio.loadmat(LF20).values())[3]
# labels21 = list(sio.loadmat(LF21).values())[3]
# labels22 = list(sio.loadmat(LF22).values())[3]
# labels23 = list(sio.loadmat(LF23).values())[3]
#
#
# EL1 = np.vstack((labels2, labels3, labels4, labels5, labels6, labels7, labels8, labels9, labels10,
#                 labels11, labels12, labels13, labels14, labels15, labels16, labels17, labels18, labels19,
#                 labels20, labels21, labels22, labels23))
#
# EL2 = np.vstack((labels1,labels3, labels4, labels5, labels6, labels7, labels8, labels9, labels10,
#                 labels11, labels12, labels13, labels14, labels15, labels16, labels17, labels18, labels19,
#                 labels20, labels21, labels22, labels23))
#
# EL3 = np.vstack((labels1, labels2, labels4, labels5, labels6, labels7, labels8, labels9, labels10,
#                 labels11, labels12, labels13, labels14, labels15, labels16, labels17, labels18, labels19,
#                 labels20, labels21, labels22, labels23))
#
# EL4 = np.vstack((labels1, labels2, labels3, labels5, labels6, labels7, labels8, labels9, labels10,
#                 labels11, labels12, labels13, labels14, labels15, labels16, labels17, labels18, labels19,
#                 labels20, labels21, labels22, labels23))
#
# EL5 = np.vstack((labels1, labels2, labels3, labels4, labels6, labels7, labels8, labels9, labels10,
#                 labels11, labels12, labels13, labels14, labels15, labels16, labels17, labels18, labels19,
#                 labels20, labels21, labels22, labels23))
#
# EL6 = np.vstack((labels1, labels2, labels3, labels4, labels5, labels7, labels8, labels9, labels10,
#                 labels11, labels12, labels13, labels14, labels15, labels16, labels17, labels18, labels19,
#                 labels20, labels21, labels22, labels23))
#
# EL7 = np.vstack((labels1, labels2, labels3, labels4, labels5, labels6, labels8, labels9, labels10,
#                 labels11, labels12, labels13, labels14, labels15, labels16, labels17, labels18, labels19,
#                 labels20, labels21, labels22, labels23))
#
# EL8 = np.vstack((labels1, labels2, labels3, labels4, labels5, labels6, labels7, labels9, labels10,
#                 labels11, labels12, labels13, labels14, labels15, labels16, labels17, labels18, labels19,
#                 labels20, labels21, labels22, labels23))
#
# EL9 = np.vstack((labels1, labels2, labels3, labels4, labels5, labels6, labels7, labels8, labels10,
#                 labels11, labels12, labels13, labels14, labels15, labels16, labels17, labels18, labels19,
#                 labels20, labels21, labels22, labels23))
#
# EL10 = np.vstack((labels1, labels2, labels3, labels4, labels5, labels6, labels7, labels8, labels9,
#                 labels11, labels12, labels13, labels14, labels15, labels16, labels17, labels18, labels19,
#                 labels20, labels21, labels22, labels23))
#
# EL11 = np.vstack((labels1, labels2, labels3, labels4, labels5, labels6, labels7, labels8, labels9, labels10,
#                 labels12, labels13, labels14, labels15, labels16, labels17, labels18, labels19,
#                 labels20, labels21, labels22, labels23))
#
# EL12 = np.vstack((labels1, labels2, labels3, labels4, labels5, labels6, labels7, labels8, labels9, labels10,
#                 labels11, labels13, labels14, labels15, labels16, labels17, labels18, labels19,
#                 labels20, labels21, labels22, labels23))
#
# EL13 = np.vstack((labels1, labels2, labels3, labels4, labels5, labels6, labels7, labels8, labels9, labels10,
#                 labels11, labels12, labels14, labels15, labels16, labels17, labels18, labels19,
#                 labels20, labels21, labels22, labels23))
#
# EL14 = np.vstack((labels1, labels2, labels3, labels4, labels5, labels6, labels7, labels8, labels9, labels10,
#                 labels11, labels12, labels13, labels15, labels16, labels17, labels18, labels19,
#                 labels20, labels21, labels22, labels23))
#
# EL15 = np.vstack((labels1, labels2, labels3, labels4, labels5, labels6, labels7, labels8, labels9, labels10,
#                 labels11, labels12, labels13, labels14, labels16, labels17, labels18, labels19,
#                 labels20, labels21, labels22, labels23))
#
# EL16 = np.vstack((labels1, labels2, labels3, labels4, labels5, labels6, labels7, labels8, labels9, labels10,
#                 labels11, labels12, labels13, labels14, labels15, labels17, labels18, labels19,
#                 labels20, labels21, labels22, labels23))
#
# EL17 = np.vstack((labels1, labels2, labels3, labels4, labels5, labels6, labels7, labels8, labels9, labels10,
#                 labels11, labels12, labels13, labels14, labels15, labels16, labels18, labels19,
#                 labels20, labels21, labels22, labels23))
#
# EL18 = np.vstack((labels1, labels2, labels3, labels4, labels5, labels6, labels7, labels8, labels9, labels10,
#                 labels11, labels12, labels13, labels14, labels15, labels16, labels17, labels19,
#                 labels20, labels21, labels22, labels23))
#
# EL19 = np.vstack((labels1, labels2, labels3, labels4, labels5, labels6, labels7, labels8, labels9, labels10,
#                 labels11, labels12, labels13, labels14, labels15, labels16, labels17, labels18,
#                 labels20, labels21, labels22, labels23))
#
# EL20 = np.vstack((labels1, labels2, labels3, labels4, labels5, labels6, labels7, labels8, labels9, labels10,
#                 labels11, labels12, labels13, labels14, labels15, labels16, labels17, labels18, labels19,
#                 labels21, labels22, labels23))
#
# EL21 = np.vstack((labels1, labels2, labels3, labels4, labels5, labels6, labels7, labels8, labels9, labels10,
#                 labels11, labels12, labels13, labels14, labels15, labels16, labels17, labels18, labels19,
#                 labels20, labels22, labels23))
#
# EL22 = np.vstack((labels1, labels2, labels3, labels4, labels5, labels6, labels7, labels8, labels9, labels10,
#                 labels11, labels12, labels13, labels14, labels15, labels16, labels17, labels18, labels19,
#                 labels20, labels21, labels23))
#
# EL23 = np.vstack((labels1, labels2, labels3, labels4, labels5, labels6, labels7, labels8, labels9, labels10,
#                 labels11, labels12, labels13, labels14, labels15, labels16, labels17, labels18, labels19,
#                 labels20, labels21, labels22))
#
# np.save("D:/Work_2/SEED-VIG/perclos_labels/label_Ex1",EL1)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/label_Ex2",EL2)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/label_Ex3",EL3)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/label_Ex4",EL4)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/label_Ex5",EL5)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/label_Ex6",EL6)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/label_Ex7",EL7)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/label_Ex8",EL8)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/label_Ex9",EL9)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/label_Ex10",EL10)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/label_Ex11",EL11)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/label_Ex12",EL12)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/label_Ex13",EL13)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/label_Ex14",EL14)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/label_Ex15",EL15)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/label_Ex16",EL16)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/label_Ex17",EL17)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/label_Ex18",EL18)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/label_Ex19",EL19)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/label_Ex20",EL20)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/label_Ex21",EL21)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/label_Ex22",EL22)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/label_Ex23",EL23)





# # 转换为.npy文件：
# # 保存为numpy数组文件（.npy文件）
# np.save("D:/Work_2/SEED-VIG/perclos_labels/labels-numpy_data.npy", L1)
# # 读取numpy文件
# # f = np.load("D:/pytorch/SEED-VIG/SEED-VIG/perclos_labels/labels-numpy_data.npy")
# # print(f.shape)
# # print(f)

# np.save("D:/Work_2/SEED-VIG/labelWMC1.npy",labels1)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/labels2.npy",labels2)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/labels3.npy",labels3)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/labels4.npy",labels4)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/labels5.npy",labels5)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/labels6.npy",labels6)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/labels7.npy",labels7)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/labels8.npy",labels8)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/labels9.npy",labels9)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/labels10.npy",labels10)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/labels11.npy",labels11)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/labels12.npy",labels12)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/labels13.npy",labels13)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/labels14.npy",labels14)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/labels15.npy",labels15)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/labels16.npy",labels16)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/labels17.npy",labels17)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/labels18.npy",labels18)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/labels19.npy",labels19)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/labels20.npy",labels20)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/labels21.npy",labels21)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/labels22.npy",labels22)
# np.save("D:/Work_2/SEED-VIG/perclos_labels/labels23.npy",labels23)



filepath = 'D:/Program Files/Polyspace/R2021a/dataset.mat'
label = sio.loadmat(filepath)['substate']
print(label.shape)
np.save("D:/Work_2/dataset2/label.npy",label)



