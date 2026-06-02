
import torch
from torch import nn
 
class LN(nn.Module):
    # 初始化
    def __init__(self, normalized_shape,  # 在哪个维度上做LN
                 eps:float = 1e-5, # 防止分母为0
                 elementwise_affine:bool = True):  # 是否使用可学习的缩放因子和偏移因子
        super(LN, self).__init__()
        # 需要对哪个维度的特征做LN, torch.size查看维度
        self.normalized_shape = normalized_shape  # [c,w*h]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        # 构造可训练的缩放因子和偏置
        if self.elementwise_affine:  
            self.gain = nn.Parameter(torch.ones(normalized_shape))  # [c,w*h]
            self.bias = nn.Parameter(torch.zeros(normalized_shape))  # [c,w*h]
 
    # 前向传播
    def forward(self, x: torch.Tensor): # [b,c,w*h]
        # 需要做LN的维度和输入特征图对应维度的shape相同
        assert self.normalized_shape == x.shape[-len(self.normalized_shape):]  # [-2:]
        # 需要做LN的维度索引
        dims = [-(i+1) for i in range(len(self.normalized_shape))]  # [b,c,w*h]维度上取[-1,-2]维度，即[c,w*h]
        # 计算特征图对应维度的均值和方差
        mean = x.mean(dim=dims, keepdims=True)  # [b,1,1]
        mean_x2 = (x**2).mean(dim=dims, keepdims=True)  # [b,1,1]
        var = mean_x2 - mean**2  # [b,c,1,1]
        x_norm = (x-mean) / torch.sqrt(var+self.eps)  # [b,c,w*h]
        # 线性变换
        if self.elementwise_affine:
            x_norm = self.gain * x_norm + self.bias  # [b,c,w*h]
        return x_norm
 
if __name__ == '__main__':
 
    x = torch.linspace(0, 23, 24, dtype=torch.float32)  # 构造输入层
    x = x.reshape([2,3,2*2])  # [b,c,w*h]
    # 实例化
    ln = LN(x.shape[1:])
    # 前向传播
    x = ln(x)
    print(x.shape)