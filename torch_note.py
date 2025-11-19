import torch 
import numpy as np
np.set_printoptions(precision=3)
#----------------------------------------
#p352
a = [1,2,3]
b = np.array([4, 5, 6], dtype=np.int32)
t_a = torch.tensor(a)
#リストからコピーを作成するので元リストは変化なし
t_b = torch.from_numpy(b)
#コピーではなく参照
print(t_a)
print(t_b)

t_ones = torch.ones(2, 3)
print(t_ones.shape)
#.shapeで何行何列か見る
print(t_ones)

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
#「平均0、標準偏差1の正規分布」からランダムな値を取り出し、形状(5,2)のテンソルを生成する

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

#--------------------------------------
#121.2.5
torch.manual_seed(1)
t = torch.rand(6)
print(t)
t_splits = torch.chunk(t,3)
#テンソルを3つに分割
print(t_splits)
print([item.numpy() for item in t_splits])


t = torch.rand(5)
print(t)
t_splits = torch.split(t, split_size_or_sections= [3,2])
print([item.numpy() for item in t_splits])
#--------------------------------------
A = torch.ones(3)
B = torch.zeros(2)
C = torch.cat([A,B], axis=0)
print(C)

A = torch.ones(3)
B = torch.zeros(3)
S = torch.stack([A, B], axis=1)
print(S)
#axis=0:０番目（行）の次元で合体
#axis=1:１番目（列）の次元で合体
#------------------------------------------
#12.3.2
#p357
torch.manual_seed(1)
t_x = torch.rand([4,3], dtype=torch.float32) #特徴慮
t_y = torch.arange(4) #クラスラベル

from torch.utils.data import Dataset 

class JointDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

        '''
self は「自分自身（今扱っているそのオブジェクト）」を指す特別な変数
もっと言うと、
自分の持っているデータ（属性）
自分が呼べる関数（メソッド）
にアクセスするために使う“鍵”。
       
        '''
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]
    #idxはデータの番号
    '''
    クラスを定義する意義は、
「データ（状態）と、それを扱う処理（機能）をひとまとめにして、
1つの“オブジェクト”として扱えるようにすること」
    '''
joint_dataset = JointDataset(t_x, t_y)

for example in joint_dataset:
    print(' x: ', example[0], ' y: ', example[1])
