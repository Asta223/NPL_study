#第六周作业：计算Bert模型参数量

import torch
import math
import torch.nn as nn
import numpy as np
from transformers import BertModel

# Bert模型 = embedding层 + transfomers层 + pool_fc层

# 加载Bert模型
model = BertModel.from_pretrained(r"..\..\bert-base-chinese",return_dict = False) #.1b
# model = BertModel.from_pretrained(r"..\..\bert-large-chinese",return_dict = False) #0.3B

# Bert默认参数
n = 2                       # 输入最大句子个数
Vocab = 21128               # 词表数量
max_sequence_length = 512   # 最大句子长度
embedding_size = 768        # embedding维度
hidden_size = 3072          # hidden层维度
num_layers = 12             # hidden层层数

# ①
# embedding层 = LAYER(token Embedding（词表） + segment Embedding（上下句） + position Embedding（句子最大长度）)
# token Embedding 参数量 = 词表大小 * embedding维度
# segment Embedding 参数量 = 上下句长度（2） * embedding维度
# position Embedding 参数量 = 句子最大长度（512） * embedding维度
embedding_parameters = (Vocab * embedding_size) \
                       + (n * embedding_size) \
                       + (max_sequence_length * embedding_size) \
                       + (embedding_size + embedding_size)

# ②
# transfomers层 = self-attention层 + feed forward层
# ATTENTION(Q,K,V) = SOFTMAX((Q * K.T)) / SQRT(D.K)) * V
# X是Embedding层输出；Q、K、V是三个线性层，其中的存在需要训练的参数；SQRT(D.K)是常数,体现多头机制,将矩阵切成若干份,以便于更加充分训练模型中的参数
# Q = X * W(Q)，形状 [L * H] * [H * H] = [L * H]
# K = X * W(K)，形状 [L * H] * [H * H] = [L * H]
# V = X * W(V)，形状 [L * H] * [H * H] = [L * H]
# (Q * K.T)) ,代表的是是Q,K两个线性层之间任意两个字之间的关系，形状 [L * H] * [H * L] = [L * L]
# SOFTMAX((Q * K.T))规划句子中每个字之间相关性的分布,解决长距离依赖
# SOFTMAX((Q * K.T) / SQRT(D.K)) * V，形状[L * L] * [L * H] = [L * H]，先计算每个字之间的相关性，再与原始输入（V）相乘，实现对原始输入更合理的调控
self_attention_parameters = (embedding_size * embedding_size + embedding_size) * 3

# OUTPUT = LINER(ATTENTION(Q,K,V)):ATTENTION过线性层,输出结果过规划层
# 残差:LayerNorm(XEmbedding + XAttention)
# LAYER_NORM层参数 = embedding_size + embedding_size
self_attention_out_parameters = (embedding_size * embedding_size + embedding_size) \
                                + (embedding_size + embedding_size)

# OUTPUT = LINER(GELU(LINER(X))):输入X,过线性层,输出的结果过激活层,再取结果过线性层,最后过规划层
# 首个线性层会将结果映射至高维(4H),以便拟合更复杂的规律;第二个线性层会将高维向量映射回低维(H)
# 残差:LayerNorm(Xforward + XAttention)
# LAYER_NORM层参数 = embedding_size + embedding_size
feed_forward_parameters = (embedding_size * hidden_size + hidden_size) \
                          + (embedding_size * hidden_size + embedding_size) \
                          + (embedding_size + embedding_size)

# ③
# pool_fc池化层
pool_fc_parameters = embedding_size * embedding_size + embedding_size

all_parameters = embedding_parameters \
                 + (self_attention_parameters + self_attention_out_parameters + feed_forward_parameters) * num_layers \
                 + pool_fc_parameters

# 输出结果: 102267648
print("Bert模型：%d" % sum(p.numel() for p in model.parameters()))
print("计算：%d" % all_parameters)
