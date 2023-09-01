#!/usr/bin/python3
# encoding: utf-8
 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
from itertools import groupby

def pcap_h2pu_imu(pack_diff_T):
    means = np.mean(pack_diff_T) # 平均值
    max = np.max(pack_diff_T)   # 最大值
    min = np.min(pack_diff_T)   # 最小值
    content = "means: {0}\nmax: {1}\nmin: {2}\n".format(means, max, min)

    step = 0.002
    for k, g in groupby(sorted(udp_head_time), key=lambda x: x//step):
        content += ('{}-{}: {}\n'.format(k*step, (k+1)*step, len(list(g))))

    plt.subplots(figsize=(12, 7))
    plt.title("CANoe Recv Imu UDP ΔT", fontdict={"fontsize": 25})  # 添加标题，调整字符大小
    plt.xlabel("udp package")  # X轴标签
    plt.ylabel("ΔT(s)")       # Y轴坐标标签
    plt.ylim(-0.005, 0.025) # y轴范围
    # plt.xlim(x1, x2) # y轴范围
    x_data = np.linspace(0, len(pack_diff_T), len(pack_diff_T))
    plt.plot(x_data, pack_diff_T, linestyle="dashed") # 设置线性

    plt.text(x=0, # 文本x轴坐标 
            y=0.025, # 文本y轴坐标
            s=content, #文本内容
            rotation=1,#文字旋转
            ha='left',#x=2.2是文字的左端位置，可选'center', 'right', 'left'
            va='top', #y=8是文字的低端位置，可选'center', 'top', 'bottom', 'baseline', 'center_baseline'
            fontdict=dict(fontsize=12, color='r',
                        family='monospace',#字体,可选'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'
                        weight='bold',#磅值，可选'light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'
                        )#字体属性设置
            )
    plt.legend() # 显示图例
    plt.savefig('pcap_h2pu_imu.jpg')
    plt.show()


def h2pu_imu_csv_process():
    data = pd.read_csv('./h2pu_imu_01.csv', skiprows=20) # 读取文件中所有数据
    h2pu_header_timestamp = data.iloc[:, 0] # 读取文件中第1列数据
    udp_recv_timestamp = data.iloc[:, 1] # 读取文件中第2列数据
    loan_timestamp = data.iloc[:, 3] # 读取文件中第4列数据
    publish_timestamp = data.iloc[:, 4] # 读取文件中第5列数据


data = pd.read_csv('./h2pu_imu_01.csv', skiprows=20) # 读取文件中所有数据
h2pu_header_timestamp = data.iloc[:, 0] # 读取文件中第1列数据
udp_recv_timestamp = data.iloc[:, 1] # 读取文件中第2列数据
loan_timestamp = data.iloc[:, 3] # 读取文件中第4列数据
publish_timestamp = data.iloc[:, 4] # 读取文件中第5列数据

diff_T1 = udp_recv_timestamp - h2pu_header_timestamp
diff_T2 = loan_timestamp - udp_recv_timestamp
diff_T3 = publish_timestamp - loan_timestamp
x_data = np.linspace(0, len(h2pu_header_timestamp), len(h2pu_header_timestamp))

plt.subplot(2, 2, 1)
plt.subplots(figsize=(12, 7))
plt.plot(x_data, diff_T1, "b--") # 设置线性
plt.title("UdpRecvT - H2puSendT", fontdict={"fontsize": 8})  # 添加标题，调整字符大小
plt.xlabel("frames")  # X轴标签
plt.ylabel("ΔT(us)")       # Y轴坐标标签

plt.subplot(2, 2, 2)
plt.plot(x_data, diff_T2, "b--") # 设置线性
plt.title("LoanT - UdpRecvT", fontdict={"fontsize": 8})  # 添加标题，调整字符大小
plt.xlabel("frames")  # X轴标签
plt.ylabel("ΔT(us)")       # Y轴坐标标签

plt.subplot(2, 2, 3)
plt.plot(x_data, diff_T3, "b--") # 设置线性
plt.title("PublishT - LoanT", fontdict={"fontsize": 8})  # 添加标题，调整字符大小
plt.xlabel("frames")  # X轴标签
plt.ylabel("ΔT(us)")  # Y轴坐标标签

# plt.subplots(figsize=(12, 7))
# plt.title("h2pu Recv ΔT", fontdict={"fontsize": 25})  # 添加标题，调整字符大小
# plt.xlabel("udp package")  # X轴标签
# plt.ylabel("ΔT(us)")       # Y轴坐标标签
# plt.ylim(-0.005, 0.025) # y轴范围
# plt.xlim(x1, x2) # y轴范围
# x_data = np.linspace(0, len(h2pu_header_timestamp), len(h2pu_header_timestamp))
# plt.plot(x_data, diff_T1, "r*") # 设置线性
# plt.plot(x_data, diff_T2, "g-") # 设置线性
# plt.plot(x_data, diff_T3, "b--") # 设置线性

# plt.text(x=0, # 文本x轴坐标 
#         y=0.025, # 文本y轴坐标
#         s=content, #文本内容
#         rotation=1,#文字旋转
#         ha='left',#x=2.2是文字的左端位置，可选'center', 'right', 'left'
#         va='top', #y=8是文字的低端位置，可选'center', 'top', 'bottom', 'baseline', 'center_baseline'
#         fontdict=dict(fontsize=12, color='r',
#                     family='monospace',#字体,可选'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace'
#                     weight='bold',#磅值，可选'light', 'normal', 'medium', 'semibold', 'bold', 'heavy', 'black'
#                     )#字体属性设置
#         )
# plt.legend() # 显示图例
plt.savefig('h2pu_imu.jpg')
plt.show()


# data = pd.read_csv('./pcap_h2pu_imu.csv') #读取文件中所有数据
# num_index = data.iloc[:, 0] # 第1列数据
# udp_head_time = data.iloc[:, 1] # 第2列数据
# pcap_h2pu_imu(udp_head_time)

# exampleFile = open('./h2pu_imu_01.csv')  # 打开csv文件
# exampleReader = csv.reader(exampleFile)  # 读取csv文件
# exampleData = list(exampleReader)  # csv数据转换为列表
# print(exampleData[][0])

# with open('./h2pu_imu_01.csv','r') as f:
#     reader = csv.reader(f)
#     h2pu_header_timestamp = [row[0] for row in reader]
#     loan_timestamp = [row[1] for row in reader] # 读取文件中第4列数据
#     publish_timestamp = [row[4] for row in reader] # 读取文件中第5列数据
# print("{}-{}-{}".format(len(h2pu_header_timestamp), len(loan_timestamp), len(publish_timestamp)))


