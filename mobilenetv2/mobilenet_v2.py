import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import math
import numpy as np

stage_out_channel = [32] + [16] + [24] * 2 + [32] * 3 + [64] * 4 + [96] * 3 + [160] * 3 + [320]
n_bit = 4

overall_channel = stage_out_channel

mid_channel = []
for i in range(len(stage_out_channel)-1):
    if i == 0:
        mid_channel += [stage_out_channel[i]]
    else:
        mid_channel += [6 * stage_out_channel[i]]

class conv2d_3x3(nn.Module):
    def __init__(self, inp, oup, stride):
        super(conv2d_3x3, self).__init__()

        self.conv1 = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(oup)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu6(out)

        return out

class conv2d_1x1(nn.Module):
    def __init__(self, inp, oup, stride):
        super(conv2d_1x1, self).__init__()

        self.conv1 = nn.Conv2d(inp, oup, 1, stride, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(oup)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu6(out)

        return out

class quan_conv2d_1x1(nn.Module):
    def __init__(self, inp, oup, stride):
        super(quan_conv2d_1x1, self).__init__()

        self.quan1 = PACT(n_bit)
        self.conv1 = HardQuantizeConv(inp, oup, n_bit, 1, stride, 0)
        self.bn1 = nn.BatchNorm2d(oup)

    def forward(self, x):

        out = self.quan1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu6(out)

        return out

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

class PACT(nn.Module):
    def __init__(self, num_bits, init_act_clip_val=2):
        super(PACT, self).__init__()
        self.num_bits = num_bits
        self.clip_val = nn.Parameter(torch.Tensor([init_act_clip_val]), requires_grad=True)

    def forward(self, x):
        x = F.relu(x)
        x = torch.where(x < self.clip_val, x, self.clip_val)
        n = float(2 ** self.num_bits - 1) / self.clip_val
        x_forward = torch.round(x * n) / n
        out = x_forward + x - x.detach()
        return out

class HardQuantizeConv(nn.Module):
    def __init__(self, in_chn, out_chn, num_bits, kernel_size=3, stride=1, padding=1, groups=1):
        super(HardQuantizeConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.num_bits = num_bits
        init_act_clip_val = 2.0
        self.clip_val = nn.Parameter(torch.Tensor([init_act_clip_val]), requires_grad=True)
        self.zero = nn.Parameter(torch.Tensor([0]), requires_grad=False)
        self.shape = (out_chn, in_chn//groups, kernel_size, kernel_size)
        self.weight = nn.Parameter((torch.rand(self.shape)-0.5) * 0.001, requires_grad=True)

    def forward(self, x):

        real_weights = self.weight
        gamma = (2**self.num_bits - 1)/(2**(self.num_bits - 1))
        scaling_factor = gamma * torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        scaled_weights = real_weights/scaling_factor
        cliped_weights = torch.where(scaled_weights < self.clip_val/2, scaled_weights, self.clip_val/2)
        cliped_weights = torch.where(cliped_weights > -self.clip_val/2, cliped_weights, -self.clip_val/2)
        n = float(2 ** self.num_bits - 1) / self.clip_val
        quan_weights_no_grad = scaling_factor * (torch.round((cliped_weights + self.clip_val/2) * n ) / n - self.clip_val/2)
        quan_weights = quan_weights_no_grad.detach() - scaled_weights.detach() + scaled_weights
        y = F.conv2d(x, quan_weights, stride=self.stride, padding=self.padding, groups=self.groups)

        return y

class LTQ(nn.Module):
    def __init__(self, num_bits):
        super(LTQ, self).__init__()
        init_range = 2.0
        self.n_val = 2 ** num_bits - 1
        self.interval = init_range / self.n_val
        self.start = nn.Parameter(torch.Tensor([0.0]), requires_grad=True)
        self.a = nn.Parameter(torch.Tensor([self.interval]* self.n_val), requires_grad=True)
        self.scale1 = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        self.scale2 = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)

        self.two =nn.Parameter(torch.Tensor([2.0]), requires_grad=False)
        self.one =nn.Parameter(torch.Tensor([1.0]), requires_grad=False)
        self.zero =nn.Parameter(torch.Tensor([0.0]), requires_grad=False)
        self.minusone = nn.Parameter(torch.Tensor([-1.0]), requires_grad=False)
        self.eps = nn.Parameter(torch.Tensor([1e-3]), requires_grad=False)

    def forward(self, x):

        x = x * self.scale1

        x_forward = x
        x_backward = x
        step_right = self.zero + 0.0

        a_pos = torch.where(self.a > self.eps, self.a, self.eps)

        for i in range(self.n_val):
            step_right += self.interval
            if i == 0:
                thre_forward = self.start + a_pos[0] / 2
                thre_backward = self.start + 0.0
                x_forward = torch.where(x > thre_forward, step_right, self.zero)
                x_backward = torch.where(x > thre_backward, self.interval/a_pos[i] * (x - thre_backward) + step_right - self.interval, self.zero)
            else:
                thre_forward += a_pos[i-1] / 2 +  a_pos[i] / 2
                thre_backward += a_pos[i-1]
                x_forward = torch.where(x > thre_forward, step_right, x_forward)
                x_backward = torch.where(x > thre_backward, self.interval/a_pos[i] * (x - thre_backward) + step_right - self.interval, x_backward)

        thre_backward += a_pos[i]
        x_backward = torch.where(x > thre_backward, self.two, x_backward)

        out = x_forward.detach() + x_backward - x_backward.detach()
        out = out * self.scale2

        return out


class bottleneck(nn.Module):
    def __init__(self, inp, oup, mid, stride):
        super(bottleneck, self).__init__()

        self.stride = stride
        self.inp = inp
        self.oup = oup

        self.bias11 = LearnableBias(inp)
        self.prelu1 = nn.PReLU(inp)
        self.bias12 = LearnableBias(inp)
        self.quan1 = LTQ(n_bit)
        self.conv1 = HardQuantizeConv(inp, mid, n_bit, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(mid)

        self.bias21 = LearnableBias(mid)
        self.prelu2 = nn.PReLU(mid)
        self.bias22 = LearnableBias(mid)
        self.quan2 = LTQ(n_bit)
        self.conv2 = HardQuantizeConv(mid, mid, n_bit, 3, stride, 1, groups=mid)
        self.bn2 = nn.BatchNorm2d(mid)

        self.bias31 = LearnableBias(mid)
        self.prelu3 = nn.PReLU(mid)
        self.bias32 = LearnableBias(mid)
        self.quan3 = LTQ(n_bit)
        self.conv3 = HardQuantizeConv(mid, oup, n_bit, 1, 1, 0)
        self.bn3 = nn.BatchNorm2d(oup)

    def forward(self, x):

        out = self.bias11(x)
        out = self.prelu1(out)
        out = self.bias12(out)
        out = self.quan1(out)
        out = self.conv1(out)
        out = self.bn1(out)

        out = self.bias21(out)
        out = self.prelu2(out)
        out = self.bias22(out)
        out = self.quan2(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = self.bias31(out)
        out = self.prelu3(out)
        out = self.bias32(out)
        out = self.quan3(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.inp == self.oup and self.stride == 1:
            return (out + x)

        else:
            return out

class MobileNetV2(nn.Module):
    def __init__(self,  input_size=224, num_classes=1000):
        super(MobileNetV2, self).__init__()

        self.feature = nn.ModuleList()

        for i in range(19):
            if i == 0:
                self.feature.append(conv2d_3x3(3, overall_channel[i], 2))
            elif i == 1:
                self.feature.append(bottleneck(overall_channel[i-1], overall_channel[i], mid_channel[i-1],1))
            elif i == 18:
                self.feature.append(quan_conv2d_1x1(overall_channel[i-1], 1280, 1))
            else:
                if stage_out_channel[i-1]!=stage_out_channel[i] and stage_out_channel[i]!=96 and stage_out_channel[i]!=320:
                    self.feature.append(bottleneck(overall_channel[i-1], overall_channel[i], mid_channel[i-1], 2))
                else:
                    self.feature.append(bottleneck(overall_channel[i-1], overall_channel[i], mid_channel[i-1], 1))


        self.pool1 = nn.AvgPool2d(7)
        self.fc = nn.Linear(1280, 1000)

    def forward(self, x):


        for i, block in enumerate(self.feature):
            if i == 0 :
                x = block(x)
            elif i == 18 :
                x = block(x)
            else :
                x = block(x)

        x = self.pool1(x)
        x = x.view(-1, 1280)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = MobileNetV1()
    print(model)
