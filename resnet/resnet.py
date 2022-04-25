import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
import torch.nn.functional as F

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50']

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def quanconv3x3(in_planes, out_planes, num_bits, stride=1):
    """3x3 convolution with padding"""
    return HardQuantizeConv(in_planes, out_planes, num_bits, kernel_size=3, stride=stride, padding=1)

def quanconv1x1(in_planes, out_planes, num_bits, stride=1):
    """1x1 convolution"""
    return HardQuantizeConv(in_planes, out_planes, num_bits, kernel_size=1, stride=stride, padding=0)

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

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

class HardQuantizeConv(nn.Module):
    def __init__(self, in_chn, out_chn, num_bits, kernel_size=3, stride=1, padding=1):
        super(HardQuantizeConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.num_bits = num_bits
        init_act_clip_val = 2.0
        self.clip_val = nn.Parameter(torch.Tensor([init_act_clip_val]), requires_grad=True)
        self.zero = nn.Parameter(torch.Tensor([0]), requires_grad=False)
        self.shape = (out_chn, in_chn, kernel_size, kernel_size)
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
        y = F.conv2d(x, quan_weights, stride=self.stride, padding=self.padding)

        return y

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        n_bit: int,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.bias11 = LearnableBias(inplanes)
        self.prelu1 = nn.PReLU(inplanes)
        self.bias12 = LearnableBias(inplanes)
        self.quan1 = LTQ(n_bit)
        self.conv1 = quanconv3x3(inplanes, planes, n_bit, stride)
        self.bn1 = norm_layer(planes)

        self.bias21 = LearnableBias(planes)
        self.prelu2 = nn.PReLU(planes)
        self.bias22 = LearnableBias(planes)
        self.quan2 = LTQ(n_bit)
        self.conv2 = quanconv3x3(planes, planes, n_bit)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.bias31 = LearnableBias(planes)
        self.prelu3 = nn.PReLU(planes)
        self.bias32 = LearnableBias(planes)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

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

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.bias31(out)
        out = self.prelu3(out)
        out = self.bias32(out)

        return out


class Bottleneck(nn.Module):

    expansion: int = 4

    def __init__(
        self,
        n_bit: int,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.bias11 = LearnableBias(inplanes)
        self.prelu1 = nn.PReLU(inplanes)
        self.bias12 = LearnableBias(inplanes)
        self.quan1 = LTQ(n_bit)
        self.conv1 = quanconv1x1(inplanes, width, n_bit)
        self.bn1 = norm_layer(width)

        self.bias21 = LearnableBias(width)
        self.prelu2 = nn.PReLU(width)
        self.bias22 = LearnableBias(width)
        self.quan2 = LTQ(n_bit)
        self.conv2 = quanconv3x3(width, width, n_bit, stride=stride)
        self.bn2 = norm_layer(width)

        self.bias31 = LearnableBias(width)
        self.prelu3 = nn.PReLU(width)
        self.bias32 = LearnableBias(width)
        self.quan3 = LTQ(n_bit)
        self.conv3 = quanconv1x1(width, planes * self.expansion, n_bit)
        self.bn3 = norm_layer(planes * self.expansion)

        self.downsample = downsample
        self.stride = stride
        self.bias01 = LearnableBias(planes * self.expansion)
        self.prelu0 = nn.PReLU(planes * self.expansion)
        self.bias02 = LearnableBias(planes * self.expansion)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

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

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.bias01(out)
        out = self.prelu0(out)
        out = self.bias02(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        n_bit: int,
        quantize_downsample: bool,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.n_bit = n_bit
        self.quantize_downsample = quantize_downsample
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.quantize_downsample:
                downsample = nn.Sequential(
                    LTQ(self.n_bit),
                    quanconv1x1(self.inplanes, planes * block.expansion, self.n_bit, stride),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.n_bit, self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.n_bit, self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    n_bit: int,
    quantize_downsample: bool,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(n_bit, quantize_downsample, block, layers, **kwargs)
    return model


def resnet18(n_bit: int, quantize_downsample: bool, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet('resnet18', n_bit, quantize_downsample, BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(n_bit: int, quantize_downsample: bool, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet('resnet34', n_bit, quantize_downsample, BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(n_bit: int, quantize_downsample: bool, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    return _resnet('resnet50', n_bit, quantize_downsample, Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

