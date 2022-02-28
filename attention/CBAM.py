import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        maxpool_out = self.maxpool(x).view(b, c)
        avgpool_out = self.avgpool(x).view(b, c)

        max_out = self.fc(maxpool_out)
        avg_out = self.fc(avgpool_out)

        out = max_out + avg_out
        out = self.sigmoid(out).view(b, c, 1, 1)

        return x * out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size, 1, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        maxpool_out, _ = torch.max(x, dim=1, keepdim=True)
        avgpool_out = torch.mean(x, dim=1, keepdim=True)
        pool_out = torch.cat([maxpool_out, avgpool_out], dim=1)
        out = self.conv(pool_out)
        out = self.sigmoid(out)

        return out


class CBAMBlock(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelAttention(channel, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        return x

if __name__ == '__main__':
    input = torch.randn(2, 512, 256, 256)

    channel_attention_model = ChannelAttention(512)
    spatial_attention_model = SpatialAttention()

    channel_attention_out = channel_attention_model(input)
    print(channel_attention_out.shape)

    spatial_attention_out = spatial_attention_model(input)
    print(spatial_attention_out.shape)

    model = CBAMBlock(512)
    output = model(input)
    print(output.shape)
