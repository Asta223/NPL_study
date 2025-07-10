# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，如果第1个数>第5个数，则为正样本，反之为负样本

"""

'''
2025/05/26 八斗人工智能-祝漳明-第二周作业：改用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类。
'''

class TorchModel(nn.Module):

    def __init__(self, input_size,hidden_size):  # ZHUZMA:level维度
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)  # 线性层

        # self.activation = torch.sigmoid  # nn.Sigmoid() sigmoid归一化函数
        # self.loss = nn.functional.mse_loss  # loss函数采用均方差损失

        self.loss = nn.functional.cross_entropy  # ZHUZMA:loss函数采用交叉熵损失，已内置了sortmax激活函数

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        # ZHUZMA:loss函数采用交叉熵损失，已内置了sortmax激活函数
        # x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        # y_pred = self.activation(x)  # (batch_size, 1) -> (batch_size, 1)
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return torch.softmax(y_pred,axis = -1)  # 输出预测结果,softmax进行规划


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# ZHUZMA:随机生成一个5维向量，返回最大值所在的索引
def build_sample(hidden_size):
    x = np.random.random(hidden_size)

    # if x[0] > x[4]:
    #     return x, 1
    # else:
    #     return x, 0

    #  ZHUZMA:查找最大值,i用于遍历，index用于记录最大值索引
    index = np.argmax(x) 
    return x,index

# 随机生成一批样本
def build_dataset(total_sample_num,hidden_size):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample(hidden_size)
        X.append(x)
        Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model,input_size):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num,input_size)
    # print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比 #y_pred为通过模型得出的预测值，y为实际样本的真实值

            # if float(y_p) < 0.5 and int(y_t) == 0:
            #     correct += 1  # 负样本判断正确
            # elif float(y_p) >= 0.5 and int(y_t) == 1:
            #     correct += 1  # 正样本判断正确
            # else:
            #     wrong += 1

            # ZHUZMA:此处改动，只要预测值与真实值相等，则预测正确

            if torch.argmax(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    print("配置参数")
    epoch_num = 50  # 训练轮数
    batch_size = 10  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    hidden_size = 5

    # 建立模型
    print("建立模型")
    model = TorchModel(input_size,hidden_size)

    # 选择优化器
    print("选择优化器")
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    print("开始训练")
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample,hidden_size)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model,input_size)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])

        #ZHUZMA:提前结束预测
        if acc > 0.98:
            print("预测拟合，结束训练")
            break

    # # 保存模型
    # torch.save(model.state_dict(), "model.bin")
    #
    # # 画图
    print("log:")
    print(log)
    # plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    # plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    # plt.legend()
    # plt.show()
    return

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)

    # model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    model.load_state_dict(torch.load(model_path, weights_only=True))  # zhuzma:只加载模型权重，不执行任意代码

    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果


if __name__ == "__main__":
    main()
    # test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.88920843],
    #             [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.00797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.09349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # predict("model.bin", test_vec)
