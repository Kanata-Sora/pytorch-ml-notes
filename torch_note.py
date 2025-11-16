import torch 
import numpy as np
np.set_printoptions(precision=3)
'''
a = [1,2,3]
b = np.array([4, 5, 6], dtype=np.int32)
t_a = torch.tensor(a)
#リストからコピーを作成するので元リストは変化なし
t_b = torch.from_numpy(b)
#コピーではなく参照
print(t_a)
print(t_b)

t_ones = torch.ones(2, 3)
t_ones.shape
print(t_ones)
'''
#------------------------------------#
#p353
t = torch.rand(3, 5)
print(t)
t_tr = torch.transpose(t, 0, 1)
#0番目（行）と1番目（列）を入れ替える。0から始まるのはプログラミングの慣習
print(t.shape, '-->', t_tr.shape)

t = torch.zeros(30)
t_reshape = t.reshape(5, 6)
print(t_reshape.shape)

#----------------------------------------

torch.manual_seed(1)
#乱数を固定
t1 = 2 * torch.rand(5, 2) -1
t2 = torch.normal(mean=0, std=1, size=(5, 2))

t3 = torch.multiply(t1, t2)
#二つの行列の積を取っているのではなく、要素ごとに掛け算を行っている
print(t3)

t4 = torch.mean(t1, axis=0)
#axis=0は行方向（縦）
print(t4)

t5 = torch.matmul(t1, torch.transpose(t2, 0, 1))
print(t5)

norm_t1 = torch.linalg.norm(t1, ord=2, dim=1)
#ord=2: ノルムの種類：L2ノルム（二乗和の平方根）**を指定
#dim=1: 列方向（横）
print(norm_t1)
print(t5)