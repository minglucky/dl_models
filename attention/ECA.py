import torch
import torch.nn as nn

import math

class ECABlock(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECABlock, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 第一种方式
        # y = self.avg_pool(x)
        # y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # y = self.sigmoid(y)

        # 第二种方式
        # y = self.avg_pool(x)
        # y =self.conv(y.squeeze(-1).permute(0,2,1))
        # y = self.sigmoid(y)
        # y = y.permute(0, 2, 1).unsqueeze(-1)

        # 第三种方式
        b, c, _, _ = x.size()
        avg = self.avg_pool(x).view(b, 1 ,c)
        out = self.conv(avg)
        out = self.sigmoid(out).view(b, c, 1, 1)

        return x * out.expand_as(x)

if __name__ == '__main__':
    model = ECABlock(512)
    input = torch.randn(2, 512, 256, 256)
    output = model(input)
    print(output.shape)