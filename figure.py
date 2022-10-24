import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
mpl.rcParams['font.size'] = 8

# labels = ['BiRNN', 'LSTM', 'BiGRU', 'BiLSTM', 'ReLU-BiLSTM','Gelu-BiLSTM']
# data1 = [70.71]
# data2 = [70.93]
# data3 = [71.48]
# data4 = [71.88]
# data5 = [72.01]
# data6 = [72.66]
#
# x = np.arange(len(labels))  # the label locations
# width = 0.35  # the width of the bars
#
# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, data1, width, label='BiRNN')
# rects2 = ax.bar(x + width/2, data2, width, label='LSTM')
# rects3 = ax.bar(x - width/2, data1, width, label='BiGRU')
# rects4 = ax.bar(x + width/2, data2, width, label='BiLSTM')
# rects5 = ax.bar(x - width/2, data1, width, label='ReLU-BiLSTM')
# rects6 = ax.bar(x + width/2, data2, width, label='Gelu-BiLSTM')
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Accuracy(%)')
# ax.set_title('Scores by group and gender')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()
# plt.show()


#TNet
# x = [1,2,3,4,5,6]
# y = [70.71,70.93,71.48,71.88,72.01,72.66]
# plt.bar(x,y,align="center",color="c",tick_label=["BiRNN", "LSTM", "BiGRU", "BiLSTM", "ReLU-BiLSTM","Gelu-BiLSTM"],hatch="/")
# # plt.scatter(x, y, s=400, marker="*", color='black',zorder=2)
#
# plt.plot(x,y,'bv--',alpha=1,linewidth=2,label='peak')
# plt.legend()
# plt.ylim(70,73)
# plt.xlabel("Methods",fontsize=10)
# plt.ylabel("Accuracy(%)",fontsize=10)
# plt.savefig("D:/Work_2/paper_figure/TNet.png")
# plt.show()


#Mydataset cross_subject
# x = np.arange(11)
# y1 = [50.69,72.08,83.69,75.31,55.45,73.23,71.79,73.67,76.01,46.56,67.89]
# y2 = [55.17,70.61,89.36,74.51,58.56,69.88,82.08,75.19,83.27,45.93,70.46]
# y3 = [63.43,75.86,70.35,78.22,50.49,70.05,80.12,71.67,85.20,60.02,70.54]
# y4 = [56.82,71.02,91.19,80.68,56.82,75.85,83.24,74.43,85.08,43.47,71.86]
#
# tick_label=["S1", "S2", "S3", "S4", "S5","S6","S7","S8","S9","S10","Avg"]
# xx = range(len(tick_label))
# plt.bar(x,y1,align="edge",width=0.2,alpha=0.7,color="brown",label="ESTCNN",hatch="")
# plt.bar(x+0.2,y2,align="edge",width=0.2,alpha=0.7,color="b",label="EEGNet",hatch="")
# plt.bar(x+0.4,y3,align="edge",width=0.2,alpha=0.7,color="darkgreen",label="Interpretable_CNN",hatch="")
# plt.bar(x+0.6,y4,align="edge",width=0.2,alpha=0.7,color="deeppink",label="CSF-GTNet",hatch="")
# plt.axhline(85.16, color='red', linestyle='--')
# plt.text(x[0],85.16,str(85.16), ha='center',va='bottom',color="black",fontsize=10)
# plt.legend()
# plt.ylim(30,100)
# plt.xlabel("Methods",fontsize=10)
# plt.ylabel("Accuracy(%)",fontsize=10)
# plt.xticks([index+0.4 for index in xx],tick_label)
# # plt.savefig("D:/Work_2/paper_figure/TNet.png")
# plt.show()



#SEED-VIG cross_subject
x = np.arange(24)
y1 = [70.21,51.36,53.63,59.91,84.23,66.39,65.32,60.66,70.03,72.36,74.01,78.20,52.31,71.35,79.30,77.96,60.33,78.55,79.20,63.74,72.65,69.63,48.62,67.82]
y2 = [76.8,58.67,52.98,67.03,76.55,62.08,70.10 ,68.91,63.61,65.98,70.32,70.39,61.52,69.28,78.39,71.10,68.49,72.91,74.43,75.16,71.38,42.85,63.17,67.49]
y3 = [73.65,62.78,61.75,61.42,70.51,67.48,64.17,58.98,70.29,64.89,78.96,65.21,61.45,80.51,84.60,72.47,61.82,81.67,77.69,78.25,75.21,60.37,59.36,69.28]
y4 = [72.43,60.95,68.59,60.58,77.28,66.59,64.30,63.72,80.34,75.36,73.44,71.15,64.06,84.13,85.86,74.16,66.59,79.09,76.16,78.97,76.32,46.27,66.35,70.99]

tick_label=["S1", "S2", "S3", "S4", "S5","S6","S7","S8","S9","S10","S11", "S12", "S13", "S14", "S15","S16","S17","S18","S19","S20","S21","S22","S23","Avg"]
xx = range(len(tick_label))
# plt.bar(x,y1,align="edge",width=0.2,alpha=0.7,color="brown",label="ESTCNN",hatch="")
# plt.bar(x+0.2,y2,align="edge",width=0.2,alpha=0.7,color="b",label="EEGNet",hatch="")
# plt.bar(x+0.4,y3,align="edge",width=0.2,alpha=0.7,color="darkgreen",label="Interpretable_CNN",hatch="")
# plt.bar(x+0.6,y4,align="edge",width=0.2,alpha=0.7,color="deeppink",label="CSF-GTNet",hatch="")
plt.plot(x,y1,'bv--',alpha=0.7,linewidth=2,label='ESTCNN',color="blue")
plt.plot(x,y2,'bo--',alpha=0.7,linewidth=2,label='EEGNet',color="darkorange")
plt.plot(x,y3,'b+--',alpha=0.7,linewidth=2,label='Interpretable_CNN',color="seagreen")
plt.plot(x,y4,'b*--',alpha=0.7,linewidth=2,label='CSF-GTNet',color="darkviolet")
plt.axhline(81.48, color='red', linestyle='--')
plt.text(x[0],81.48,str(81.48), ha='center',va='bottom',color="black",fontsize=10)
plt.legend()
plt.ylim(30,100)
plt.xlabel("Methods",fontsize=10)
plt.ylabel("Accuracy(%)",fontsize=10)
plt.xticks([index for index in xx],tick_label)
# plt.savefig("D:/Work_2/paper_figure/TNet.png")
plt.show()