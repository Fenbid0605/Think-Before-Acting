import torch
from torch import nn
from torch.nn import functional as F


class NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlockND, self).__init__()

        self.sub_sample = sub_sample  # 是否进行下采样

        self.in_channels = in_channels  # 输入通道数
        self.inter_channels = inter_channels  # 中间通道数

        # 如果未指定中间通道数，默认为输入通道数的一半
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # 定义 g、theta、phi 的卷积层
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        # 定义 W 层，可选择是否使用批归一化0
        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                          kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.in_channels)
            )
            # nn.init.constant(self.W[1].weight, 0)  # 初始化权重
            # nn.init.constant(self.W[1].bias, 0)    # 初始化偏置
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                               kernel_size=1, stride=1, padding=0)
            # nn.init.constant(self.W.weight, 0)  # 初始化权重
            # nn.init.constant(self.W.bias, 0)    # 初始化偏置

        # 定义 theta 和 phi 的卷积层
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        # 如果进行下采样，则对 g 和 phi 添加池化层
        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, x):
        '''
        前向传播方法
        :param x: 输入张量，形状为 (b, c, t, h, w) （对于3D数据）
        :return: 输出张量，形状与输入相同
        '''
        batch_size = x.size(0)  # 获取批量大小

        # 计算 g(x)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # 变形为 (b, inter_channels, N)
        g_x = g_x.permute(0, 2, 1)  # 调整维度顺序

        # 计算 theta(x) 和 phi(x)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)  # 调整维度顺序
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        # 计算注意力权重
        f = torch.matmul(theta_x, phi_x)  # 矩阵乘法
        f_div_C = F.softmax(f, dim=-1)  # 归一化

        # 加权聚合
        y = torch.matmul(f_div_C, g_x)  # 使用权重对 g_x 加权
        y = y.permute(0, 2, 1).contiguous()  # 调整维度顺序
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])  # 变形为 (b, inter_channels, t, h, w)

        # 融合输入和输出
        W_y = self.W(y)  # 通过 W 层
        z = W_y + x  # 残差连接

        return z  # 返回最终输出


if __name__ == '__main__':
    model = NonLocalBlockND(256)
    # 生成一个1x256x80x80的张量
    data = torch.randn(1, 256, 80, 80)
    updata = model.forward(data)
    print(updata.shape)  # 输出张量的形状