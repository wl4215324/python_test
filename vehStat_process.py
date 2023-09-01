#!/usr/bin/python3
# encoding: utf-8
 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
from itertools import groupby

data = pd.read_csv('./VehStat.csv', header=None) # 读取文件中所有数据, 无表头
rolling_cnt = data.iloc[:,len(data.columns)-5] # 读取文件中倒数第5列数据rolling counter
send_time_stamp = data.iloc[:,len(data.columns)-4] # 读取文件中倒数第4列数据,发送时间戳
recv_time_stamp = data.iloc[:,len(data.columns)-1] # 读取文件中倒数第1列数据,接收时间戳
transmit_delay = recv_time_stamp - send_time_stamp # 传输延时

rolling_cnt_diff = rolling_cnt.diff() # rolling counter前后帧差值
means = np.mean(rolling_cnt_diff) # 平均值
max = np.max(rolling_cnt_diff)    # 最大值
min = np.min(rolling_cnt_diff)    # 最小值
content = "means: {0}\nmax: {1}\nmin: {2}\ncount: {3}\n".format(means, max, min, len(rolling_cnt_diff))
x_data = np.linspace(0, len(rolling_cnt_diff), len(rolling_cnt_diff))

fig, ax1 = plt.subplots(2, 2, figsize=(11, 7))
plt.title("0830 19:30 ~ 0901 08:30 mqttlib Desktop testing ", fontdict={"fontsize": 8})  # 添加标题，调整字符大小
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.4)

ax1[0][0].plot(x_data, rolling_cnt_diff, "b--") # 设置线性
ax1[0][0].text(x=170000, # 文本x轴坐标 
         y=1.05, # 文本y轴坐标
         s=content, #文本内容
         rotation=0,#文字旋转
         ha='left',#x=2.2是文字的左端位置，可选'center', 'right', 'left'
         va='top', #y=8是文字的低端位置，可选'center', 'top', 'bottom', 'baseline', 'center_baseline'
         fontdict=dict(fontsize=7, color='r',
                    family='monospace',#字体,可选'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'
                    weight='bold',#磅值，可选'light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'
                    )#字体属性设置
        )

ax1[0][0].set_title("Mqtt Recv rolling counter diff", fontdict={"fontsize": 7})  # 添加标题，调整字符大小
ax1[0][0].set_xlabel("frames", fontdict={"fontsize": 7})       # X轴标签
ax1[0][0].set_ylabel("ΔrollingCnt", fontdict={"fontsize": 7})  # Y轴坐标标签

send_time_stamp_diff = send_time_stamp.diff() # rolling counter前后帧差值
means = np.mean(send_time_stamp_diff) # 平均值
max = np.max(send_time_stamp_diff)    # 最大值
min = np.min(send_time_stamp_diff)    # 最小值
content = "means: {0}\nmax: {1}\nmin: {2}\ncount: {3}\n".format(means, max, min, len(send_time_stamp_diff))
# 按区间分组统计各组个数,区间长度50
step = 50
for k, g in groupby(sorted(send_time_stamp_diff), key=lambda x: x//step):
    content += ('{}-{}: {}\n'.format(k*step, (k+1)*step - 1, len(list(g))))

x_data = np.linspace(0, len(send_time_stamp_diff), len(send_time_stamp_diff))

ax1[0][1].plot(x_data, send_time_stamp_diff, "b*") # 设置线性
ax1[0][1].text(x=170000, # 文本x轴坐标 
         y=245, # 文本y轴坐标
         s=content, #文本内容
         rotation=0,#文字旋转
         ha='left',#x=2.2是文字的左端位置，可选'center', 'right', 'left'
         va='top', #y=8是文字的低端位置，可选'center', 'top', 'bottom', 'baseline', 'center_baseline'
         fontdict=dict(fontsize=7, color='r',
                    family='monospace',#字体,可选'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'
                    weight='bold',#磅值，可选'light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'
                    )#字体属性设置
        )
ax1[0][1].set_title("send timeStamp diff", fontdict={"fontsize": 7})  # 添加标题，调整字符大小
ax1[0][1].set_xlabel("frames", fontdict={"fontsize": 7})  # X轴标签
ax1[0][1].set_ylabel("ΔT(ms)", fontdict={"fontsize": 7})  # Y轴坐标标签

recv_time_stamp_diff = recv_time_stamp.diff() # rolling counter前后帧差值
means = np.mean(recv_time_stamp_diff) # 平均值
max = np.max(recv_time_stamp_diff)    # 最大值
min = np.min(recv_time_stamp_diff)    # 最小值
content = "means: {0}\nmax: {1}\nmin: {2}\ncount: {3}\n".format(means, max, min, len(recv_time_stamp_diff))
# 按区间分组统计各组个数,区间长度50
step = 50
for k, g in groupby(sorted(recv_time_stamp_diff), key=lambda x: x//step):
    content += ('{}-{}: {}\n'.format(k*step, (k+1)*step - 1, len(list(g))))
    
x_data = np.linspace(0, len(recv_time_stamp_diff), len(recv_time_stamp_diff))
ax1[1][0].plot(x_data, recv_time_stamp_diff, "b*") # 设置线性
ax1[1][0].text(x=170000, # 文本x轴坐标 
         y=650, # 文本y轴坐标
         s=content, #文本内容
         rotation=0,#文字旋转
         ha='left',#x=2.2是文字的左端位置，可选'center', 'right', 'left'
         va='top', #y=8是文字的低端位置，可选'center', 'top', 'bottom', 'baseline', 'center_baseline'
         fontdict=dict(fontsize=7, color='r',
                    family='monospace',#字体,可选'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'
                    weight='bold',#磅值，可选'light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'
                    )#字体属性设置
        )
ax1[1][0].set_title("recv timeStamp diff", fontdict={"fontsize": 7})  # 添加标题，调整字符大小
ax1[1][0].set_xlabel("frames", fontdict={"fontsize": 7})  # X轴标签
ax1[1][0].set_ylabel("ΔT(ms)", fontdict={"fontsize": 7})  # Y轴坐标标签

means = np.mean(transmit_delay) # 平均值
max = np.max(transmit_delay)    # 最大值
min = np.min(transmit_delay)    # 最小值
content = "means: {0}\nmax: {1}\nmin: {2}\ncount: {3}\n".format(means, max, min, len(transmit_delay))
# 按区间分组统计各组个数,区间长度50
step = 50
for k, g in groupby(sorted(transmit_delay), key=lambda x: x//step):
    content += ('{}-{}: {}\n'.format(k*step, (k+1)*step - 1, len(list(g))))

x_data = np.linspace(0, len(transmit_delay), len(transmit_delay))

ax1[1][1].plot(x_data, transmit_delay, "g--") # 设置线性
ax1[1][1].text(x=210000, # 文本x轴坐标 
         y=530, # 文本y轴坐标
         s=content, #文本内容
         rotation=0,#文字旋转
         ha='left',#x=2.2是文字的左端位置，可选'center', 'right', 'left'
         va='top', #y=8是文字的低端位置，可选'center', 'top', 'bottom', 'baseline', 'center_baseline'
         fontdict=dict(fontsize=7, color='r',
                    family='monospace',#字体,可选'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'
                    weight='bold',#磅值，可选'light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'
                    )#字体属性设置
        )
ax1[1][1].set_title("recv send delay", fontdict={"fontsize": 7})  # 添加标题，调整字符大小
ax1[1][1].set_xlabel("frames", fontdict={"fontsize": 7})  # X轴标签
ax1[1][1].set_ylabel("ΔT(ms)", fontdict={"fontsize": 7})  # Y轴坐标标签

plt.savefig("vehStat.jpg")
plt.show()



