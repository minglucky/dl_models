import torch
import torch.nn as nn
from functools import reduce

# arXiv(Paper): https://arxiv.org/pdf/1903.06586.pdf
# github(Official): https://github.com/implus/SKNet
# github(pytorch): https://github.com/pppLang/SKNet

# github(pytorch): https://github.com/pppLang/SKNet
class SKConv(nn.Module):
    def __init__(self, features, WH, M, G, r, stride=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3 + i * 2, stride=stride, padding=1 + i, groups=G),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))
        # self.gap = nn.AvgPool2d(int(WH/stride))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


# https://zhuanlan.zhihu.com/p/76033612
class SKLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, M=2, r=16, L=32):
        '''
        :param in_channels:  ??????????????????
        :param out_channels: ??????????????????   ???????????? ??????????????????????????????
        :param stride:  ??????????????????1
        :param M:  ?????????
        :param r: ??????Z???????????????????????????d ?????????????????????????????? ??????S->Z ??????????????????????????? ??????????????????
        :param L:  ?????????????????????Z?????????????????????32
        '''
        super(SKLayer, self).__init__()
        d = max(in_channels // r, L)  # ????????????Z ?????????d
        self.M = M
        self.out_channels = out_channels
        self.conv = nn.ModuleList()  # ?????????????????? ?????? ????????????????????????
        for i in range(M):
            # ?????????????????????????????? ????????????5x5??? ???3X3???dilation=2??????????????? ???????????????????????????G=32
            self.conv.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, padding=1 + i, dilation=1 + i, groups=32, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)))
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # ?????????pool???????????????    ???????????????1????????? GAP
        self.fc1 = nn.Sequential(nn.Conv2d(out_channels, d, 1, bias=False),
                                 nn.BatchNorm2d(d),
                                 nn.ReLU(inplace=True))  # ??????
        self.fc2 = nn.Conv2d(d, out_channels * M, 1, 1, bias=False)  # ??????
        self.softmax = nn.Softmax(dim=1)  # ??????dim=1  ??????????????????????????????????????????softmax,?????? ????????????a+b+..=1

    def forward(self, input):
        batch_size = input.size(0)
        output = []
        # the part of split
        for i, conv in enumerate(self.conv):
            # print(i,conv(input).size())
            output.append(conv(input))
        # the part of fusion
        U = reduce(lambda x, y: x + y, output)  # ????????????????????? ????????????U
        s = self.global_pool(U)
        z = self.fc1(s)  # S->Z??????
        a_b = self.fc2(z)  # Z->a???b ??????  ????????????conv 1x1????????????????????????????????????????????????a,????????????b
        a_b = a_b.reshape(batch_size, self.M, self.out_channels, -1)  # ????????????????????? ????????????????????????
        a_b = self.softmax(a_b)  # ??????????????????????????????????????????softmax
        # the part of selection
        a_b = list(a_b.chunk(self.M, dim=1))  # split to a and b   chunk???pytorch????????????tensor??????????????????????????? ??????tensor???
        a_b = list(map(lambda x: x.reshape(batch_size, self.out_channels, 1, 1), a_b))  # ???????????????  ??????????????????????????????
        V = list(map(lambda x, y: x * y, output, a_b))  # ???????????????  ????????????????????????U ???????????????
        V = reduce(lambda x, y: x + y, V)  # ???????????????????????? ???????????????
        return V


if __name__ == '__main__':
    # model = SKConv(512, 32, 2, 8, 2)
    model = SKLayer(512, 512)
    input = torch.randn(2, 512, 256, 256)
    output = model(input)
    print(output.shape)
