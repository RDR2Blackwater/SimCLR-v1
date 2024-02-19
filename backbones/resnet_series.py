import torch
from torch import Tensor
import torch.nn as nn

from typing import Optional, Callable, List, Tuple


# Pointwise & Depthwise Implement
def pointwise_conv(in_channels: int, out_channels: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    return nn.Conv2d(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=1,
                     stride=stride,
                     padding=0,
                     dilation=dilation,
                     groups=groups,
                     bias=False
                     )


def depthwise_conv(in_channels: int, out_channels: int, stride: int, kernel_size: int = 3, dilation: int = 1):
    return nn.Conv2d(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=dilation,
                     dilation=dilation,
                     groups=out_channels,
                     bias=False
                     )


# Bottleneck Implement
class BottleneckResidual(nn.Module):
    def __init__(self, in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 stride: int = 1,
                 kernel_size: int = 3,
                 downsample: Optional[nn.Module] = None,
                 activation=nn.ReLU(inplace=True),
                 norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(BottleneckResidual, self).__init__()

        self.conv1 = pointwise_conv(in_channels, hidden_channels)
        self.norm_1 = norm_layer(hidden_channels)
        # Replace the 3*3 conv with depth-wise conv
        # reduce 50% of parameters compared to original ResNet-50 when the width multiplier >= 2
        self.conv2 = depthwise_conv(hidden_channels, hidden_channels, stride, kernel_size)
        self.norm_2 = norm_layer(hidden_channels)
        self.conv3 = pointwise_conv(hidden_channels, out_channels)
        self.norm_3 = norm_layer(out_channels)

        self.downsample = downsample
        self.activation = nn.ReLU() if activation is None else activation

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.conv1(x)
        out = self.norm_1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.norm_2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.norm_3(out)

        if self.downsample is not None:
            residual = self.downsample(residual)
        out += residual
        out = self.activation(out)

        return out


# Basic implement of ResNet50
class ResNet50(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 make_blocks: Tuple = (3, 4, 6, 3)):
        super(ResNet50, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.fc = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Construct Layers
        layers: List[nn.Module] = []
        # Input Layer is a 7*7 Conv with Stride=2
        layers.append(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3))
        layers.append(norm_layer(64))
        # Apply a 3*3 Maxpool with stride=2, padding=1
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        # Make hidden blocks
        # conv2 requires downsampling, but it just need a bigger channel width
        downsampling = nn.Sequential(
            pointwise_conv(64, 256, 1),
            norm_layer(256)
        )
        layers.append(BottleneckResidual(64, 64, 256, downsample=downsampling, norm_layer=norm_layer))
        for _ in range(1, make_blocks[0]):
            layers.append(BottleneckResidual(256, 64, 256, norm_layer=norm_layer))
        # conv3, conv4, and conv5
        for conv_index, blocks in enumerate(make_blocks[1:], start=1):
            # Downsampling for conv3_1, conv4_1, and conv5_1
            in_channels, hidden_channels, out_channels = 64 * (2 ** conv_index), \
                                                         64 * (2 ** conv_index), \
                                                         256 * (2 ** conv_index)
            downsampling = nn.Sequential(
                pointwise_conv(in_channels*2, out_channels, 2),
                norm_layer(out_channels)
            )
            layers.append(
                BottleneckResidual(in_channels*2, hidden_channels, out_channels, stride=2, downsample=downsampling, norm_layer=norm_layer))
            # Hidden Layers
            for _ in range(1, blocks):
                layers.append(BottleneckResidual(out_channels, hidden_channels, out_channels, norm_layer=norm_layer))
        # Average Pool & Linear will be applied in forward
        self.resnet = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:      # Ignore dim0 (batch_size)
        out_resnet = self.resnet(x)           # [2048, 7, 7] (If input pixel is 224*224)
        out_avg = self.avgpool(out_resnet)       # [2048, 1, 1]
        out_flatten = torch.flatten(out_avg, 1)  # [2048]
        out = self.fc(out_flatten)               # [10]
        return out


# this is a modified version of ResNet50, will be applied in SimCLR
# For SimCLR, wider & deeper network is better
class SimCLR_ResNet50(nn.Module):
    def __init__(self,
                 num_classes: int = 1000,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 make_blocks: Tuple = (3, 4, 6, 3),
                 width_multiplier: int = 1):
        super(SimCLR_ResNet50, self).__init__()
        if norm_layer is None:
            self.norm_layer = nn.BatchNorm2d

        # Following paper, the last fc layer would be abandoned, the whole network is working as an encoder
        # The fc will be rewritten in pretrain and fine-tune parse
        self.fc = nn.Linear(2048 * width_multiplier, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Construct Layers
        layers: List[nn.Module] = []
        # 7*7 Conv
        self.conv1 = nn.Conv2d(3, 64 * width_multiplier, kernel_size=7, stride=2, padding=3)
        self.bn1 = self.norm_layer(64 * width_multiplier)
        # Apply a 3*3 Maxpool with Stride=2
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Make hidden blocks
        # conv2 requires a bigger channel width, so the stride of "downsampling" is 1
        downsampling = nn.Sequential(
            pointwise_conv(64 * width_multiplier, 256 * width_multiplier, 1),
            self.norm_layer(256 * width_multiplier)
        )
        layers.append(self._make_layers(0, make_blocks[0], width_multiplier, downsampling))
        # conv3 ~ conv5
        for conv_index, blocks in enumerate(make_blocks[1:], start=1):
            # Downsample
            in_channels, hidden_channels, out_channels = 64 * (2 ** conv_index) * width_multiplier, \
                                                         64 * (2 ** conv_index) * width_multiplier, \
                                                         256 * (2 ** conv_index) * width_multiplier
            downsampling = nn.Sequential(
                pointwise_conv(in_channels*2, out_channels, 2),
                self.norm_layer(out_channels)
            )
            layers.append(self._make_layers(conv_index, blocks, width_multiplier, downsampling))

        self.resnet = nn.Sequential(*layers)

    def _make_layers(self, conv_index: int, blocks: int, width_multiplier, downsampling) -> nn.Sequential:
        layers_seq: List[nn.Module] = []
        in_channels, hidden_channels, out_channels = 64 * (2 ** conv_index) * width_multiplier, \
                                                     64 * (2 ** conv_index) * width_multiplier, \
                                                     256 * (2 ** conv_index) * width_multiplier
        # Downsample
        # conv2 requires downsample, but it just need a bigger channel width and the stride=1
        layers_seq.append(BottleneckResidual(in_channels if conv_index == 0 else in_channels * 2,
                                             hidden_channels,
                                             out_channels,
                                             stride=1 if conv_index == 0 else 2,
                                             downsample=downsampling,
                                             norm_layer=self.norm_layer))
        # Hidden Layers
        for _ in range(1, blocks):
            layers_seq.append(BottleneckResidual(out_channels,
                                                 hidden_channels,
                                                 out_channels,
                                                 norm_layer=self.norm_layer))

        return nn.Sequential(*layers_seq)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        out = self.resnet(x)
        out_avg = self.avgpool(out)              # [batch_size, 2048, 1, 1] (If width_multiplier=1)
        out_flatten = torch.flatten(out_avg, 1)  # [batch_size, 2048] dim=1 represents save the dim of channel
        out = self.fc(out_flatten)
        return out

    """
        def __init__(self, width_multiplier: int, num_classes: int = 1000):
            super(SimCLR_ResNet50, self).__init__()

            # Load original ResNet50
            resnet50 = models.resnet50(weights=None)
            # res = models.resnet._resnet(Bottleneck, [3, 4, 6, 3], None, True)

            # 分解ResNet50的各个层
            self.conv1 = resnet50.conv1
            self.bn1 = resnet50.bn1
            self.relu = resnet50.relu
            self.maxpool = resnet50.maxpool
            self.layer1 = self._modify_layer(resnet50.layer1, width_multiplier, first_layer=True)
            self.layer2 = self._modify_layer(resnet50.layer2, width_multiplier)
            self.layer3 = self._modify_layer(resnet50.layer3, width_multiplier)
            self.layer4 = self._modify_layer(resnet50.layer4, width_multiplier)

            # 新的全局平均池化层和分类层
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(resnet50.fc.in_features, num_classes)

        def _modify_layer(self, layer, multiplier, first_layer=False):
            modified_layer = []

            for idx, module in enumerate(layer.children()):
                if isinstance(module, nn.Conv2d):
                    # 修改Conv2d层的输入和输出通道数
                    module.in_channels *= multiplier
                    module.out_channels *= multiplier

                    # 对第一个残差块进行降采样
                    if first_layer and idx == 0:
                        downsample = nn.Sequential(
                            nn.Conv2d(module.in_channels // 2, module.out_channels, kernel_size=1, stride=2, bias=False),
                            nn.BatchNorm2d(module.out_channels)
                        )
                        modified_layer.extend([downsample, module])
                    else:
                        modified_layer.append(module)

                elif isinstance(module, nn.BatchNorm2d):
                    # 修改BatchNorm2d层的特征通道数
                    module.num_features *= multiplier
                    modified_layer.append(module)
                elif isinstance(module, nn.Sequential):
                    # 递归修改Sequential内部的层
                    modified_layer.extend(self._modify_layer(submodule, multiplier) for submodule in module)
                else:
                    modified_layer.append(module)

            return nn.Sequential(*modified_layer)

        def forward(self, x):
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

        """