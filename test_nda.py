'''
non-local dual attention
'''
from test_network import BasicConv2d
import torch
import torch.nn as nn

class NDA_pam(nn.Module):
    '''
    ppmcc based position attention module
    '''
    def __init__(self, in_channel):
        super(NDA_pam, self).__init__()
        self.query_conv = BasicConv2d(in_channel, in_channel // 8, 1)
        self.key_conv = BasicConv2d(in_channel, in_channel // 8, 1)
        self.value_conv = BasicConv2d(in_channel, in_channel // 8, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.size()
        # modify sort
        q = self.query_conv(x).view(b, -1, h*w).permute(0, 2, 1)
        k = self.key_conv(x).view(b, -1, h*w)
        q_mean = torch.mean(q, dim=1, keepdim=True)
        k_mean = torch.mean(k, dim=2, keepdim=True)
        q_centered = q - q_mean
        k_centered = k - k_mean
        cov_qk = torch.matmul(q_centered, k_centered) / (q.shape[1] - 1)
        q_std = torch.std(q, dim=1, unbiased=False)
        k_std = torch.std(k, dim=2, unbiased=False)
        ppmcc = cov_qk / (q_std * k_std)
        # attention map
        att_map = self.softmax(ppmcc).permute(0, 2, 1)
        v = self.value_conv(x).view(b, -1, h*w)
        out = torch.matmul(v, att_map).view(b, c, h, w)
        # add to x
        out = self.gamma * out + x
        return out

class NDA_cam(nn.Module):
    '''
    efficient channel attention module
    '''
    def __init__(self, in_channel, kernel_size):
        super(NDA_cam, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.eca_conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)