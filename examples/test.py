import numpy as np
import faiss
import random
from random import randint
# #
# # # 结合文本向量示例
# # sentences = ["cat sits on mat", "dog plays fetch", "bird sings in tree"]
# # embeddings = np.array([
# #     [0.2, 0.3, 0.4, 0.5],
# #     [0.5, 0.2, 0.1, 0.0],
# #     [0.1, 0.4, 0.5, 0.2]
# # ]).astype('float32')
# #
# # # 创建索引
# # index = faiss.IndexFlatL2(embeddings.shape[1])
# # index.add(embeddings)
# #
# # # 查询相似句子
# # query_embedding = np.array([[0.3, 0.3, 0.4, 0.4]]).astype('float32')
# # distances, indices = index.search(query_embedding, 2)
# #
# # print(indices)
# # print(distances)
# #
# # # 显示结果
# # print("最相似的 2 个句子:")
# # indices_list = indices[0]
# # for i in range(len(indices_list)):
# #     print(f"- {sentences[indices_list[i]]} (距离: {distances[0][i]:.4f})")
#
# data = np.random.rand(5, 3).astype('float32')
# print(type(data))
# # random.seed(0)
# index_it = randint(0,min(20,30))
# print(index_it)

# import pandas as pd
#
# df = pd.DataFrame({
#     'numbers': [1, 2, 3],
#     'date': ['2023-01-05', '2023-03-25', '2023-01-24']
# })
#
# date = '2023-01-05'
#
# # 0     True
# # 1    False
# # 2    False
# # Name: date, dtype: bool
# df['date_match']=df['date'].apply(lambda x: x == date)
# print(df[df['date_match']==True])
# import torch
# from torch.nn.functional import selu_
#
# x = torch.tensor([[0.2370, 1.7276,7276],[0.70, 1.276,1.232]])
# y = torch.tensor([[0.2370, 1.4276,7276],[0.20, 1.26,1.22]])
# sigmoid = torch.nn.Sigmoid()
# prd = torch.nn.Linear(3,1)
# pred = prd(x)
# print(pred)
# output = sigmoid(pred).view(-1)
# print(output)
# y = torch.transpose(y,-2,-1)
# print(x)
# print(y)
# z = torch.matmul(x,y)
# print(z)



# A = torch.tensor([[1, 2], [3, 4]])  # 2x2
# B = torch.tensor([[5, 6], [7, 8]])  # 2x2
# matmul_result = torch.matmul(A, B)  # 或使用 A @ B
# print(matmul_result)
# 输出: [[1 * 5+2 * 7, 1 * 6+2 * 8], [3 * 5+4 * 7, 3 * 6+4 * 8]] = [[19, 22], [43, 50]]


import torch

# 创建一个形状为 (1, 3, 1, 4) 的张量
# x = torch.randn(128, 2, 4)
# print(f"原始形状: {x.shape}")  # torch.Size([1, 3, 1, 4])
#
# # 移除所有大小为1的维度
# y = x.squeeze()
# print(f"squeeze后: {y.shape}")  # torch.Size([3, 4])
#
# # 指定移除特定维度
# x = x.transpose(-1,-2)
# x = x.mean(dim=-1)
# x = x.unsqueeze(1)
# print(x.shape)
# # z = x.squeeze(2)  # 只移除第0维
# # print(f"squeeze(0)后: {z.shape}")  # torch.Size([3, 1, 4])

# def generate_numbers(target_N):
#     import random
#     numbers = [target_N//4 for _ in range(4)]  # 先生成3个随机数，最后一个由前三个决定
#     total = sum(numbers)
#     last_number = target_N - total  # 计算最后一个数的值使其总和为10
#     numbers[-1] += last_number
#     return numbers
#
# print(generate_numbers(30))

a= np.ones((1,4,16))
print(a.shape)
b = np.squeeze(a,0)
print(b.shape)
c = np.mean(b,axis=0)
print(c.shape)
print(c)
d = np.expand_dims(c,axis=0)
print(d)
print(d.shape)