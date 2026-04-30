"""
member2_model/models.py
=======================
Member 2 — Model Architectures
Implements ALL eight CNN architectures used in Peterson et al. (ICCV 2019):

  1. ResNet-110          resnet_basic_110
  2. ResNet-20           lightweight variant
  3. WRN-28-10           wrn_28_10
  4. VGG-16-BN           vgg_15_BN_64  (BN variant)
  5. ResNet-preact-164   resnet_preact_bottleneck_164
  6. ResNeXt-29-8x64d    resnext_29_8x64d
  7. DenseNet-BC-100-12  densenet_BC_100_12
  8. PyramidNet-110-270  pyramidnet_basic_110_270
  9. Shake-Shake-26      shake_shake_26_2x64d  (without cutout — data aug)
 10. LightCNN            tiny debug model (not in paper, for quick testing)

All models:
  - Accept 32×32 RGB CIFAR inputs
  - Output 10-class raw logits (no softmax)
  - Use AdaptiveAvgPool so spatial size is never hard-coded
  - Are accessed via get_model(name, device)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ══════════════════════════════════════════════════════════════════════════════
# 1.  RESNET  (resnet_basic_110  and  resnet20)
# ══════════════════════════════════════════════════════════════════════════════
# He et al., "Deep Residual Learning for Image Recognition", CVPR 2016.
#
# CIFAR design:
#   stem  : 3×3 conv, 16 ch, stride 1  (image stays 32×32, no max-pool)
#   stage1: n × BasicBlock, 16 ch, 32×32
#   stage2: n × BasicBlock, 32 ch, 16×16  (stride-2 first block)
#   stage3: n × BasicBlock, 64 ch,  8×8   (stride-2 first block)
#   head  : GlobalAvgPool → FC(10)
#
#   depth = 6n + 2  →  n=3 → ResNet-20,  n=18 → ResNet-110
# ══════════════════════════════════════════════════════════════════════════════

class BasicBlock(nn.Module):
    """
    Standard residual block with two 3×3 convs and a skip connection.

    Skip connection uses a 1×1 conv to match dimensions when
    stride > 1 or channel count changes.
    """
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.shortcut(x))
        return out


class ResNetCIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_ch = 16
        self.conv1  = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(16)
        self.layer1 = self._make_stage(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_stage(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_stage(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(64 * block.expansion, num_classes)
        self._init_weights()

    def _make_stage(self, block, out_ch, n, stride):
        layers = [block(self.in_ch, out_ch, stride)]
        self.in_ch = out_ch * block.expansion
        for _ in range(n - 1):
            layers.append(block(self.in_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def resnet20(num_classes=10):
    """ResNet-20: depth=6×3+2=20. ~272K params. Fast debug/baseline."""
    return ResNetCIFAR(BasicBlock, [3, 3, 3], num_classes)

def resnet_basic_110(num_classes=10):
    """ResNet-110: depth=6×18+2=110. ~1.7M params. Paper model."""
    return ResNetCIFAR(BasicBlock, [18, 18, 18], num_classes)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  RESNET PRE-ACTIVATION  (resnet_preact_bottleneck_164)
# ══════════════════════════════════════════════════════════════════════════════
# He et al., "Identity Mappings in Deep Residual Networks", ECCV 2016.
#
# Key difference from standard ResNet:
#   Standard ResNet:    Conv → BN → ReLU → Conv → BN → (+) → ReLU
#   Pre-activation:     BN → ReLU → Conv → BN → ReLU → Conv → (+)
#
# Why this matters:
#   The skip connection is now a pure identity — no activation sits on it.
#   Gradients flow back completely unmodified through the skip path.
#   This makes very deep networks (164 layers) train more stably.
#
# Bottleneck block (used at depth ≥ 164):
#   1×1 conv (reduce channels) → 3×3 conv → 1×1 conv (expand channels)
#   expansion = 4  →  internal width = out_ch // 4
#
# CIFAR-164 config: 3 stages × 18 bottleneck blocks + 2 = 164 layers
# ══════════════════════════════════════════════════════════════════════════════

class PreActBottleneck(nn.Module):
    """
    Pre-activation bottleneck block.

    Structure (pre-act order):
      BN→ReLU→Conv(1×1) → BN→ReLU→Conv(3×3) → BN→ReLU→Conv(1×1) → (+skip)

    The bottleneck squeezes channels by 4× internally:
      in_ch  →  out_ch//4  →  out_ch//4  →  out_ch
    """
    expansion = 4

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        mid_ch = out_ch // self.expansion   # bottleneck width

        # Pre-act: BN+ReLU sit BEFORE the conv, not after
        self.bn1   = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)

        self.bn2   = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride,
                               padding=1, bias=False)

        self.bn3   = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)

        # Shortcut — only a linear projection, no BN/ReLU on the skip path
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1,
                                      stride=stride, bias=False)

    def forward(self, x):
        # Note: BN+ReLU come BEFORE each conv
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out = out + self.shortcut(x)   # pure addition, no activation here
        return out


class ResNetPreAct(nn.Module):
    """
    Pre-activation ResNet for CIFAR.
    Paper uses depth=164 with bottleneck blocks → 3 stages × 18 blocks.
    """
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_ch = 16

        self.conv1  = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_stage(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_stage(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_stage(block, 256, num_blocks[2], stride=2)

        # Final BN+ReLU before pooling (pre-act networks need this)
        self.bn_final = nn.BatchNorm2d(256)
        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))
        self.fc       = nn.Linear(256, num_classes)

        self._init_weights()

    def _make_stage(self, block, out_ch, n, stride):
        layers = [block(self.in_ch, out_ch, stride)]
        self.in_ch = out_ch
        for _ in range(n - 1):
            layers.append(block(self.in_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)              # no BN/ReLU here — pre-act blocks handle it
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn_final(out)) # final BN+ReLU before pooling
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def resnet_preact_bottleneck_164(num_classes=10):
    """
    ResNet-preact-164: 3 stages × 18 bottleneck blocks × (3 convs each) + 2
                       = 164 layers.  ~1.7M params.
    """
    return ResNetPreAct(PreActBottleneck, [18, 18, 18], num_classes)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  WIDE RESNET  (wrn_28_10)
# ══════════════════════════════════════════════════════════════════════════════
# Zagoruyko & Komodakis, "Wide Residual Networks", BMVC 2016.
#
# Insight: wider networks (more filters) learn faster than deep ones.
# WRN-28-10 means depth=28, width_factor=10.
# Channel counts: [16, 16×10, 32×10, 64×10] = [16, 160, 320, 640]
#
# Block uses pre-activation + dropout between the two 3×3 convs:
#   BN→ReLU→Conv→Dropout→BN→ReLU→Conv  (+skip)
# ══════════════════════════════════════════════════════════════════════════════

class WideBasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.0):
        super().__init__()
        self.bn1     = nn.BatchNorm2d(in_ch)
        self.conv1   = nn.Conv2d(in_ch, out_ch, 3, stride=stride,
                                 padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.bn2     = nn.BatchNorm2d(out_ch)
        self.conv2   = nn.Conv2d(out_ch, out_ch, 3, stride=1,
                                 padding=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1,
                                      stride=stride, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + self.shortcut(x)


class WideResNet(nn.Module):
    def __init__(self, depth=28, width=10, dropout=0.3, num_classes=10):
        super().__init__()
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        ch = [16, 16 * width, 32 * width, 64 * width]

        self.conv1  = nn.Conv2d(3, ch[0], 3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_stage(ch[0], ch[1], n, stride=1, dropout=dropout)
        self.layer2 = self._make_stage(ch[1], ch[2], n, stride=2, dropout=dropout)
        self.layer3 = self._make_stage(ch[2], ch[3], n, stride=2, dropout=dropout)
        self.bn_final = nn.BatchNorm2d(ch[3])
        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))
        self.fc       = nn.Linear(ch[3], num_classes)
        self._init_weights()

    @staticmethod
    def _make_stage(in_ch, out_ch, n, stride, dropout):
        layers = [WideBasicBlock(in_ch, out_ch, stride, dropout)]
        for _ in range(n - 1):
            layers.append(WideBasicBlock(out_ch, out_ch, 1, dropout))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn_final(out))
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def wrn_28_10(num_classes=10):
    """WRN-28-10: depth=28, width=10, dropout=0.3. ~36.5M params."""
    return WideResNet(depth=28, width=10, dropout=0.3, num_classes=num_classes)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  VGG-16 WITH BATCH NORMALISATION  (vgg_15_BN_64)
# ══════════════════════════════════════════════════════════════════════════════
# Simonyan & Zisserman, "Very Deep Convolutional Networks", ICLR 2015.
#
# Adapted for CIFAR (32×32):
#   - Same conv block pattern as original VGG-16
#   - BatchNorm after every conv (stabilises training, allows higher LR)
#   - AdaptiveAvgPool(1,1) replaces the large FC head (handles 32×32 input)
#   - Two FC layers → 10 classes
# ══════════════════════════════════════════════════════════════════════════════

VGG16_CONFIG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
                512, 512, 512, 'M', 512, 512, 512, 'M']

class VGG(nn.Module):
    def __init__(self, config=VGG16_CONFIG, num_classes=10, dropout=0.5):
        super().__init__()
        self.features   = self._make_features(config)
        self.avgpool    = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        self._init_weights()

    @staticmethod
    def _make_features(config):
        layers, in_ch = [], 3
        for v in config:
            if v == 'M':
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers += [nn.Conv2d(in_ch, v, 3, padding=1, bias=False),
                           nn.BatchNorm2d(v),
                           nn.ReLU(inplace=True)]
                in_ch = v
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.classifier(out)


def vgg16_bn(num_classes=10):
    """VGG-16 with BatchNorm. ~14.7M params."""
    return VGG(VGG16_CONFIG, num_classes=num_classes)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  RESNEXT-29-8x64d  (resnext_29_8x64d)
# ══════════════════════════════════════════════════════════════════════════════
# Xie et al., "Aggregated Residual Transformations for Deep Neural Networks",
# CVPR 2017.
#
# Core idea — "cardinality":
#   Instead of one wide 3×3 conv, use C parallel (grouped) convolutions,
#   each operating on a narrow slice, then sum the results.
#   This gives more diverse feature transformations for the same parameters.
#
#   cardinality C = 8   (number of parallel groups)
#   base_width    = 64  (channels per group)
#   → total bottleneck width = C × base_width = 8 × 64 = 512
#      but scaled by stage → 128 / 256 / 512 for stages 1/2/3
#
# Block structure (bottleneck with grouped conv):
#   1×1 conv (expand to C×d channels)
#   3×3 grouped conv (groups=C, each group handles d channels)
#   1×1 conv (project back)
#   + skip connection
#
# CIFAR-29 depth: 3 stages × 3 blocks × 3 convs + 2 = 29
# ══════════════════════════════════════════════════════════════════════════════

class ResNeXtBlock(nn.Module):
    """
    ResNeXt bottleneck block with grouped (aggregated) convolutions.

    Parameters
    ----------
    in_ch       : input channels
    out_ch      : output channels (after expansion)
    stride      : spatial stride
    cardinality : number of parallel groups  (C=8 in paper)
    base_width  : channels per group at the first stage (64 in paper)
    stage_idx   : which stage (1,2,3) — scales base_width accordingly
    """
    expansion = 2   # output channels = mid_ch × expansion

    def __init__(self, in_ch, out_ch, stride=1,
                 cardinality=8, base_width=64, stage_idx=1):
        super().__init__()

        # Scale width by stage (doubles each stage for CIFAR config)
        width = int(base_width * (2 ** (stage_idx - 1)))
        mid_ch = cardinality * width   # total grouped conv channels

        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid_ch)

        # Grouped convolution: C separate convolutions, each on width channels
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride,
                               padding=1, groups=cardinality, bias=False)
        self.bn2   = nn.BatchNorm2d(mid_ch)

        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = F.relu(out + self.shortcut(x))
        return out


class ResNeXt(nn.Module):
    """
    ResNeXt-29 8×64d for CIFAR.
    depth=29: 3 stages × 3 blocks × 3 convs + 2 = 29
    """
    def __init__(self, cardinality=8, base_width=64,
                 num_blocks=None, num_classes=10):
        super().__init__()
        if num_blocks is None:
            num_blocks = [3, 3, 3]

        # Stage output channels: 256, 512, 1024
        channels = [256, 512, 1024]

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.in_ch = 64

        self.layer1 = self._make_stage(channels[0], num_blocks[0], 1,
                                       cardinality, base_width, stage_idx=1)
        self.layer2 = self._make_stage(channels[1], num_blocks[1], 2,
                                       cardinality, base_width, stage_idx=2)
        self.layer3 = self._make_stage(channels[2], num_blocks[2], 2,
                                       cardinality, base_width, stage_idx=3)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(channels[2], num_classes)
        self._init_weights()

    def _make_stage(self, out_ch, n, stride, cardinality, base_width, stage_idx):
        layers = [ResNeXtBlock(self.in_ch, out_ch, stride,
                               cardinality, base_width, stage_idx)]
        self.in_ch = out_ch
        for _ in range(n - 1):
            layers.append(ResNeXtBlock(self.in_ch, out_ch, 1,
                                       cardinality, base_width, stage_idx))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def resnext_29_8x64d(num_classes=10):
    """ResNeXt-29 8×64d. ~34.4M params. Paper model."""
    return ResNeXt(cardinality=8, base_width=64,
                   num_blocks=[3, 3, 3], num_classes=num_classes)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  DENSENET-BC-100-12  (densenet_BC_100_12)
# ══════════════════════════════════════════════════════════════════════════════
# Huang et al., "Densely Connected Convolutional Networks", CVPR 2017.
#
# Core idea — dense connections:
#   Every layer receives feature maps from ALL preceding layers as input.
#   This maximally reuses features, reduces parameters, and strengthens
#   gradient flow to early layers.
#
#   Growth rate k=12: each layer adds exactly 12 new feature maps.
#   After n layers in a dense block, the block outputs k₀ + n×k channels
#   where k₀ is the number of input channels.
#
# BC variant (Bottleneck + Compression):
#   Bottleneck: 1×1 conv reduces to 4k channels before the 3×3 conv
#               (saves computation while keeping the dense connection benefit)
#   Compression: transition layers halve the channel count (θ=0.5)
#               (keeps the model compact)
#
# depth=100, k=12:
#   3 dense blocks, each with (100-4)/3/2 = 16 bottleneck layers
#   separated by transition layers → total 100 conv layers
# ══════════════════════════════════════════════════════════════════════════════

class DenseLayer(nn.Module):
    """
    Single dense layer (bottleneck variant):
      BN→ReLU→Conv(1×1, 4k ch) → BN→ReLU→Conv(3×3, k ch)

    Input is the concatenation of ALL previous layers in the block.
    Output is k new feature maps (growth rate).
    """
    def __init__(self, in_ch, growth_rate, bn_size=4, dropout=0.0):
        super().__init__()
        inter_ch = bn_size * growth_rate   # 4k channels in bottleneck

        self.bn1   = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, inter_ch, 1, bias=False)   # bottleneck

        self.bn2   = nn.BatchNorm2d(inter_ch)
        self.conv2 = nn.Conv2d(inter_ch, growth_rate, 3,
                               padding=1, bias=False)             # output k ch

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x may be a list of tensors (all previous outputs) — concatenate first
        if isinstance(x, list):
            x = torch.cat(x, dim=1)
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.dropout(out)
        return out


class DenseBlock(nn.Module):
    """
    A dense block: n layers where each layer's input is concatenation
    of all previous layers' outputs (including original input).

    After n layers: output channels = in_ch + n × growth_rate
    """
    def __init__(self, num_layers, in_ch, growth_rate, bn_size=4, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # Each layer sees in_ch + i×growth_rate input channels
            self.layers.append(
                DenseLayer(in_ch + i * growth_rate, growth_rate,
                           bn_size, dropout)
            )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feat = layer(features)         # pass full list
            features.append(new_feat)
        return torch.cat(features, dim=1)      # concatenate all outputs


class TransitionLayer(nn.Module):
    """
    Transition between dense blocks:
      BN→ReLU→Conv(1×1, θ×in_ch)→AvgPool(2×2)
    Compression factor θ=0.5 halves the channel count.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.bn   = nn.BatchNorm2d(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(F.relu(self.bn(x))))


class DenseNet(nn.Module):
    """
    DenseNet-BC for CIFAR.
    depth=100, growth_rate=12, compression=0.5 → DenseNet-BC-100-12.

    Parameters
    ----------
    depth       : total layers (must be 3n+4 for 3 blocks)
    growth_rate : k, new channels per dense layer
    compression : θ, channel reduction at transitions (0.5 = BC variant)
    dropout     : dropout inside dense layers
    """
    def __init__(self, depth=100, growth_rate=12, compression=0.5,
                 dropout=0.0, num_classes=10):
        super().__init__()

        assert (depth - 4) % 6 == 0, "DenseNet depth must be 6n+4"
        n_layers = (depth - 4) // 6   # layers per dense block (bottleneck)
        # With bottleneck, each 'layer' is 2 convs, so paper counts differently
        # depth=100, BC: n = (100-4)/(3*2) = 16 layers per block

        n = (depth - 4) // (3 * 2)    # bottleneck layers per block

        # Initial conv: 2×growth_rate channels, keeps 32×32
        init_ch = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, init_ch, 3, stride=1, padding=1, bias=False)

        # ── Dense Block 1 ─────────────────────────────────────────────────────
        in_ch = init_ch
        self.block1 = DenseBlock(n, in_ch, growth_rate, dropout=dropout)
        in_ch = in_ch + n * growth_rate
        out_ch = int(in_ch * compression)
        self.trans1 = TransitionLayer(in_ch, out_ch)
        in_ch = out_ch

        # ── Dense Block 2 ─────────────────────────────────────────────────────
        self.block2 = DenseBlock(n, in_ch, growth_rate, dropout=dropout)
        in_ch = in_ch + n * growth_rate
        out_ch = int(in_ch * compression)
        self.trans2 = TransitionLayer(in_ch, out_ch)
        in_ch = out_ch

        # ── Dense Block 3 ─────────────────────────────────────────────────────
        self.block3 = DenseBlock(n, in_ch, growth_rate, dropout=dropout)
        in_ch = in_ch + n * growth_rate

        # ── Head ──────────────────────────────────────────────────────────────
        self.bn_final = nn.BatchNorm2d(in_ch)
        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))
        self.fc       = nn.Linear(in_ch, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = F.relu(self.bn_final(out))
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def densenet_bc_100_12(num_classes=10):
    """DenseNet-BC-100-12: depth=100, k=12, θ=0.5. ~0.77M params."""
    return DenseNet(depth=100, growth_rate=12, compression=0.5,
                    num_classes=num_classes)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  PYRAMIDNET-110-270  (pyramidnet_basic_110_270)
# ══════════════════════════════════════════════════════════════════════════════
# Han et al., "Deep Pyramidal Residual Networks", CVPR 2017.
#
# Core idea — gradual channel widening:
#   Standard ResNet doubles channels abruptly at stage boundaries.
#   PyramidNet spreads the channel increase evenly across EVERY layer,
#   like a pyramid.  This avoids the sudden capacity jumps and gives
#   smoother feature learning.
#
#   At layer k (out of N total), the channel count is:
#     ch(k) = 16 + round(α × k / N)
#   where α = 270 (the "widening factor")
#   At the last layer: 16 + 270 = 286 channels
#
# Uses BasicBlock (two 3×3 convs) with pre-activation BN.
# depth=110: 3 stages × 18 BasicBlocks + 2 = 110 layers.
# Stride-2 is applied at the first block of stages 2 and 3.
# ══════════════════════════════════════════════════════════════════════════════

class PyramidBlock(nn.Module):
    """
    PyramidNet basic block.
    Pre-activation BN, no ReLU on the shortcut path.
    The shortcut uses zero-padding to match the widened channel count
    (avoids the 1×1 conv projection cost).
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1,
                               padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_ch)   # extra BN before addition

        self.stride   = stride
        self.in_ch    = in_ch
        self.out_ch   = out_ch

        # Shortcut: AvgPool for stride, zero-pad for channel expansion
        self.shortcut_pool = nn.AvgPool2d(2, 2) if stride == 2 else None

    def forward(self, x):
        out = self.conv1(self.bn1(x))          # BN → Conv  (no ReLU before first conv)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.bn3(out)                    # BN before addition

        # Shortcut with zero-padding for extra channels
        sc = x
        if self.shortcut_pool is not None:
            sc = self.shortcut_pool(sc)
        if self.in_ch != self.out_ch:
            # Pad extra channels with zeros on the channel dimension
            pad = torch.zeros(sc.size(0),
                              self.out_ch - self.in_ch,
                              sc.size(2), sc.size(3),
                              device=sc.device, dtype=sc.dtype)
            sc = torch.cat([sc, pad], dim=1)

        return F.relu(out + sc)


class PyramidNet(nn.Module):
    """
    PyramidNet for CIFAR.
    depth=110, alpha=270 (additive widening).
    """
    def __init__(self, depth=110, alpha=270, num_classes=10):
        super().__init__()
        assert (depth - 2) % 6 == 0
        n = (depth - 2) // 6   # blocks per stage (18 for depth=110)

        # Compute the channel at each block using the pyramid formula
        # Total blocks = 3 × n
        total_blocks = 3 * n
        # ch[k] = 16 + round(alpha * (k+1) / total_blocks)
        channels = [16] + [
            16 + round(alpha * (k + 1) / total_blocks)
            for k in range(total_blocks)
        ]

        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)

        layers = []
        for stage in range(3):
            for block_idx in range(n):
                global_idx = stage * n + block_idx
                in_ch  = channels[global_idx]
                out_ch = channels[global_idx + 1]
                # Stride-2 at the start of stages 2 and 3 (not stage 1)
                stride = 2 if (stage > 0 and block_idx == 0) else 1
                layers.append(PyramidBlock(in_ch, out_ch, stride))

        self.layers  = nn.Sequential(*layers)
        final_ch     = channels[-1]

        self.bn_final = nn.BatchNorm2d(final_ch)
        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))
        self.fc       = nn.Linear(final_ch, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn_final(out))
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def pyramidnet_110_270(num_classes=10):
    """PyramidNet-110-270: depth=110, alpha=270. ~28.3M params."""
    return PyramidNet(depth=110, alpha=270, num_classes=num_classes)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  SHAKE-SHAKE  (shake_shake_26_2x64d)
# ══════════════════════════════════════════════════════════════════════════════
# Gastaldi, "Shake-Shake Regularization", arXiv 2017.
#
# Core idea — stochastic branch mixing:
#   Each residual block has TWO parallel branches (both compute features).
#   During TRAINING:
#     forward pass  — sum branches with random weight α:  α·b1 + (1-α)·b2
#     backward pass — use a DIFFERENT random weight β:    β·b1 + (1-β)·b2
#   This creates a strong regularisation effect — the network can't rely
#   on either branch alone because the mixing is randomised independently
#   for forward and backward.
#   During TESTING: use equal weights (0.5·b1 + 0.5·b2).
#
# Architecture: 26-layer, 2 branches, 64 base channels per branch (2×64d)
#   3 stages × 4 ShakeShake blocks + 2 = 26 layers
#   Each block has 2 branches of 2 conv layers each
#
# Note on Cutout:
#   The paper's best result uses cutout data augmentation at test time.
#   Cutout is a DATA AUGMENTATION technique (random square masking during
#   training), NOT part of the model architecture itself. It belongs in
#   Member 1's dataset/transform pipeline. The model here is pure
#   architecture. Member 3 can optionally enable it in the transforms.
# ══════════════════════════════════════════════════════════════════════════════

class ShakeShakeBranch(nn.Module):
    """One branch of a Shake-Shake block: BN→ReLU→Conv→BN→ReLU→Conv."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return out


class ShakeShakeBlock(nn.Module):
    """
    Shake-Shake residual block with two stochastic branches.

    Training:
      forward  uses α ~ Uniform(0,1)  to blend branches
      backward uses β ~ Uniform(0,1)  independently

    Testing:
      uses fixed weight 0.5 (deterministic average of both branches)
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.branch1 = ShakeShakeBranch(in_ch, out_ch, stride)
        self.branch2 = ShakeShakeBranch(in_ch, out_ch, stride)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(stride, stride),
                # Pad channels with zeros (cheaper than 1×1 conv)
            )
        self.in_ch  = in_ch
        self.out_ch = out_ch
        self.stride = stride

    def _shake(self, b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
        """
        Apply shake-shake mixing.
        Training: random α for forward, random β for backward.
        Testing : fixed 0.5.
        """
        if not self.training:
            return 0.5 * (b1 + b2)

        # α: per-sample random scalar in (0,1)
        batch = b1.size(0)
        alpha = torch.rand(batch, 1, 1, 1, device=b1.device, dtype=b1.dtype)
        beta  = torch.rand(batch, 1, 1, 1, device=b1.device, dtype=b1.dtype)

        # Forward: α·b1 + (1-α)·b2
        # Backward: β·b1 + (1-β)·b2  — achieved via the shake trick
        # We detach both branches and add back with different weights
        out = (beta  - alpha).detach() * b1 + \
              (alpha - beta ).detach() * b2 + \
              alpha * b1 + (1 - alpha) * b2
        # Simplified: equals alpha*b1 + (1-alpha)*b2 in forward,
        #             but beta *b1 + (1-beta )*b2 in backward
        return out

    def forward(self, x):
        b1  = self.branch1(x)
        b2  = self.branch2(x)
        out = self._shake(b1, b2)

        # Shortcut with channel zero-padding if needed
        sc = x
        if self.stride > 1:
            sc = F.avg_pool2d(sc, self.stride, self.stride)
        if self.in_ch != self.out_ch:
            pad = torch.zeros(sc.size(0), self.out_ch - self.in_ch,
                              sc.size(2), sc.size(3),
                              device=sc.device, dtype=sc.dtype)
            sc = torch.cat([sc, pad], dim=1)

        return out + sc


class ShakeShake(nn.Module):
    """
    Shake-Shake-26 2×64d for CIFAR.
    26 layers: 3 stages × 4 ShakeShake blocks (each block = 2 branches × 2 convs)
               + 2 = 26 layers.
    Base width = 64 channels (doubled at each stage: 64→128→256).
    """
    def __init__(self, base_channels=64, num_blocks=None, num_classes=10):
        super().__init__()
        if num_blocks is None:
            num_blocks = [4, 4, 4]

        # Stage channel widths
        channels = [base_channels, base_channels * 2, base_channels * 4]

        self.conv1  = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(16)
        self.in_ch  = 16

        self.layer1 = self._make_stage(channels[0], num_blocks[0], stride=1)
        self.layer2 = self._make_stage(channels[1], num_blocks[1], stride=2)
        self.layer3 = self._make_stage(channels[2], num_blocks[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(channels[2], num_classes)
        self._init_weights()

    def _make_stage(self, out_ch, n, stride):
        layers = [ShakeShakeBlock(self.in_ch, out_ch, stride)]
        self.in_ch = out_ch
        for _ in range(n - 1):
            layers.append(ShakeShakeBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def shake_shake_26_2x64d(num_classes=10):
    """Shake-Shake-26 2×64d. ~26M params. State-of-the-art in paper."""
    return ShakeShake(base_channels=64, num_blocks=[4, 4, 4],
                      num_classes=num_classes)


# ══════════════════════════════════════════════════════════════════════════════
# 9.  LIGHTCNN  —  fast debug model  (not in paper)
# ══════════════════════════════════════════════════════════════════════════════
# Tiny 3-block CNN. NOT used for real experiments.
# Purpose: verify the full pipeline (data→model→loss→backprop) works in
# seconds before committing to a multi-hour training run.
# ══════════════════════════════════════════════════════════════════════════════

class LightCNN(nn.Module):
    """~95K params. Trains in seconds on CPU."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,  32, 3, padding=1, bias=False), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


# ══════════════════════════════════════════════════════════════════════════════
# 10. FACTORY  —  get_model(name, device)
# ══════════════════════════════════════════════════════════════════════════════
# All 8 paper models + lightcnn registered here.
# Member 3 and Member 4 only need to call get_model("name", device).
# ══════════════════════════════════════════════════════════════════════════════

_MODEL_REGISTRY = {
    # Paper models (Section 5.1 identifiers → our constructors)
    "resnet_basic_110"       : resnet_basic_110,           # resnet_basic_110
    "resnet20"        : resnet20,            # lightweight baseline
    "wrn28_10"        : wrn_28_10,           # wrn_28_10
    "vgg15_bn_64"           : vgg16_bn,            # vgg_15_BN_64
    "resnet_preact_bottleneck_164"   : resnet_preact_bottleneck_164,   # resnet_preact_bottleneck_164
    "resnext_29_8x64d"         : resnext_29_8x64d,    # resnext_29_8x64d
    "densenet_bc_100_12"        : densenet_bc_100_12,  # densenet_BC_100_12
    "pyramidnet_basic_110_270"      : pyramidnet_110_270,  # pyramidnet_basic_110_270
    "shakeshake"      : shake_shake_26_2x64d,# shake_shake_26_2x64d
    # Debug
    "lightcnn"        : LightCNN,
}


def get_model(name: str, device: torch.device,
              num_classes: int = 10) -> nn.Module:
    """
    Model factory. Returns the requested architecture on `device`.

    Parameters
    ----------
    name        : model name string (see _MODEL_REGISTRY keys above)
    device      : torch.device('cpu') or torch.device('cuda')
    num_classes : output logits (default 10)

    Returns
    -------
    nn.Module ready for training / evaluation.
    """
    key = name.lower().strip()
    if key not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'.\n"
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )
    model = _MODEL_REGISTRY[key](num_classes=num_classes)
    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    """Return trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(name: str, device: torch.device) -> None:
    """Build model, run a dummy forward pass, print shape and param count."""
    model  = get_model(name, device)
    n_params = count_parameters(model)
    dummy  = torch.zeros(2, 3, 32, 32, device=device)
    model.eval()
    with torch.no_grad():
        out = model(dummy)
    print(f"[{name:<35s}]  {n_params:>12,}   {str(tuple(dummy.shape)):>20s}  {str(tuple(out.shape))}")


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST  (python models.py)
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    print("Model Summary  (all 8 paper models + debug)")
    print("=" * 90)
    print(f"  {'Model':<35s}  {'Params':>12s}  {'Input':>20s}  {'Output'}")
    for name in _MODEL_REGISTRY:
        print_model_summary(name, device)
    print("=" * 90)
    print("\nAll models built and forward-pass verified successfully.")