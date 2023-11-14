'''
Skip Texture Enhanced Pooling Pyramid
'''
from test_network import BasicConv2d, DepthwiseSeparableConv2d
import torch
import torch.nn as nn
class Stepp(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Stepp, self).__init__()
        self.relu = nn.ReLU(True)
        self.up_m = nn.Sequential(
            nn.AdaptiveAvgPool2d(128),
            BasicConv2d(in_channel, out_channel, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.down_m_0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            DepthwiseSeparableConv2d(out_channel, out_channel, 3, padding=1, dilation=1),
        )
        self.down_m_1 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            DepthwiseSeparableConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.down_m_2 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            DepthwiseSeparableConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.down_m_3 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            DepthwiseSeparableConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)

    def forward(self, x):
        up_res = self.up_m(x)
        print(up_res.shape)
        down_res_0 = self.down_m_0(x)
        print(down_res_0.shape)
        down_res_1 = self.down_m_1(down_res_0)
        print(down_res_1.shape)
        down_res_2 = self.down_m_2(down_res_1)
        print(down_res_2.shape)
        down_res_3 = self.down_m_3(down_res_2)
        print(down_res_3.shape)
        x_cat = self.conv_cat(torch.cat((down_res_0, down_res_1, down_res_2, down_res_3), 1))

        output = self.relu(x_cat + up_res)
        return output