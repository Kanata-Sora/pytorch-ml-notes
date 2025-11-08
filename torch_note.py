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
