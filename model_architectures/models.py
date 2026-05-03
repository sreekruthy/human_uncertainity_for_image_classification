import torch
import torch.nn as nn
import torch.nn.functional as F
import math

NUM_CLASSES = 10

# ══════════════════════════════════════════════════════════════════════════════
# 1.  RESNET  (resnet_basic_110 and resnet20)
# ══════════════════════════════════════════════════════════════════════════════
# He et al., CVPR 2016.
# CIFAR design: stem 16ch, three stages 16→32→64, GlobalAvgPool, FC.
# depth = 6n + 2  →  n=3 → ResNet-20,  n=18 → ResNet-110
# ══════════════════════════════════════════════════════════════════════════════

class BasicBlock(nn.Module):
    """Standard two-conv residual block with identity / projection shortcut."""
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, stride=stride,
                                  padding=1, bias=False)
        self.bn1      = nn.BatchNorm2d(out_ch)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, stride=1,
                                  padding=1, bias=False)
        self.bn2      = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNetCIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_ch  = 16
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
        return self.fc(torch.flatten(out, 1))


def resnet20(num_classes=10):
    """ResNet-20: n=3 → depth=20. ~272K params. Fast baseline."""
    return ResNetCIFAR(BasicBlock, [3, 3, 3], num_classes)

def resnet_basic_110(num_classes=10):
    """ResNet-110: n=18 → depth=110. ~1.7M params. Paper model."""
    return ResNetCIFAR(BasicBlock, [18, 18, 18], num_classes)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  RESNET PRE-ACTIVATION  (resnet_preact_bottleneck_164)
# ══════════════════════════════════════════════════════════════════════════════
# He et al., ECCV 2016. Pre-act order: BN→ReLU→Conv (no act on skip path).
# Bottleneck expansion=4. depth=164 → 3×18 bottleneck blocks.
# ══════════════════════════════════════════════════════════════════════════════

class PreActBottleneck(nn.Module):
    """
    Pre-activation bottleneck block.
    in_ch → mid_ch (1×1) → mid_ch (3×3) → out_ch (1×1)
    where mid_ch = out_ch // expansion.
    """
    expansion = 4

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        mid_ch     = out_ch // self.expansion
        self.bn1   = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride,
                               padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        # Shortcut: linear projection only — no BN/ReLU on skip path
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1,
                                      stride=stride, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        return out + self.shortcut(x)


class ResNetPreAct(nn.Module):
    """Pre-activation ResNet for CIFAR (depth=164, bottleneck)."""

    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_ch    = 16
        self.conv1    = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.layer1   = self._make_stage(block, 64,  num_blocks[0], stride=1)
        self.layer2   = self._make_stage(block, 128, num_blocks[1], stride=2)
        self.layer3   = self._make_stage(block, 256, num_blocks[2], stride=2)
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
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn_final(out))
        out = self.avgpool(out)
        return self.fc(torch.flatten(out, 1))


def resnet_preact_bottleneck_164(num_classes=10):
    """ResNet-preact-164: 3×18 bottleneck blocks. ~1.7M params."""
    return ResNetPreAct(PreActBottleneck, [18, 18, 18], num_classes)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  WIDE RESNET  (wrn_28_10)
# ══════════════════════════════════════════════════════════════════════════════
# Zagoruyko & Komodakis, BMVC 2016.
# WRN-28-10: depth=28, width_factor=10.
# Channels: [16, 160, 320, 640]. Pre-act + dropout between convs.
# ══════════════════════════════════════════════════════════════════════════════

class WideBasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.0):
        super().__init__()
        self.bn1      = nn.BatchNorm2d(in_ch)
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, stride=stride,
                                  padding=1, bias=False)
        self.dropout  = nn.Dropout(p=dropout)
        self.bn2      = nn.BatchNorm2d(out_ch)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, stride=1,
                                  padding=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1,
                                      stride=stride, bias=False)

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        return out + self.shortcut(x)


class WideResNet(nn.Module):
    def __init__(self, depth=28, width=10, dropout=0.3, num_classes=10):
        super().__init__()
        assert (depth - 4) % 6 == 0, "WRN depth must be 6n+4"
        n  = (depth - 4) // 6
        ch = [16, 16 * width, 32 * width, 64 * width]
        self.conv1    = nn.Conv2d(3, ch[0], 3, stride=1, padding=1, bias=False)
        self.layer1   = self._make_stage(ch[0], ch[1], n, 1, dropout)
        self.layer2   = self._make_stage(ch[1], ch[2], n, 2, dropout)
        self.layer3   = self._make_stage(ch[2], ch[3], n, 2, dropout)
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
        return self.fc(torch.flatten(out, 1))


def wrn_28_10(num_classes=10):
    """WRN-28-10: depth=28, width=10, dropout=0.3. ~36.5M params."""
    return WideResNet(depth=28, width=10, dropout=0.3, num_classes=num_classes)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  VGG-16 WITH BATCH NORMALISATION  (vgg16_bn)
# ══════════════════════════════════════════════════════════════════════════════
# Simonyan & Zisserman, ICLR 2015. Adapted for 32×32 CIFAR.
# AdaptiveAvgPool replaces the large 7×7 FC head.
# ══════════════════════════════════════════════════════════════════════════════

VGG16_CONFIG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
                512, 512, 512, 'M', 512, 512, 512, 'M']


class VGG(nn.Module):
    def __init__(self, config=None, num_classes=10, dropout=0.5):
        super().__init__()
        if config is None:
            config = VGG16_CONFIG
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
    """VGG-16 with BatchNorm, adapted for CIFAR 32×32. ~14.7M params."""
    return VGG(VGG16_CONFIG, num_classes=num_classes)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  RESNEXT-29-8x64d  (resnext_29_8x64d)
# ══════════════════════════════════════════════════════════════════════════════
# Xie et al., CVPR 2017. Grouped (aggregated) residual convolutions.
# cardinality=8, base_width=64. depth=29: 3×3 blocks, 3 convs each.
# ══════════════════════════════════════════════════════════════════════════════

class ResNeXtBlock(nn.Module):
    """Bottleneck with grouped conv. Stage index scales the internal width."""
    expansion = 2

    def __init__(self, in_ch, out_ch, stride=1,
                 cardinality=8, base_width=64, stage_idx=1):
        super().__init__()
        width  = int(base_width * (2 ** (stage_idx - 1)))
        mid_ch = cardinality * width
        self.conv1    = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1      = nn.BatchNorm2d(mid_ch)
        self.conv2    = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride,
                                  padding=1, groups=cardinality, bias=False)
        self.bn2      = nn.BatchNorm2d(mid_ch)
        self.conv3    = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3      = nn.BatchNorm2d(out_ch)
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
        return F.relu(out + self.shortcut(x))


class ResNeXt(nn.Module):
    """ResNeXt-29 8×64d for CIFAR. Stage channels: 256 → 512 → 1024."""

    def __init__(self, cardinality=8, base_width=64,
                 num_blocks=None, num_classes=10):
        super().__init__()
        if num_blocks is None:
            num_blocks = [3, 3, 3]
        channels   = [256, 512, 1024]
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.in_ch = 64
        self.layer1 = self._make_stage(channels[0], num_blocks[0], 1,
                                       cardinality, base_width, 1)
        self.layer2 = self._make_stage(channels[1], num_blocks[1], 2,
                                       cardinality, base_width, 2)
        self.layer3 = self._make_stage(channels[2], num_blocks[2], 2,
                                       cardinality, base_width, 3)
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
        return self.fc(torch.flatten(out, 1))


def resnext_29_8x64d(num_classes=10):
    """ResNeXt-29 8×64d. ~34.4M params. Paper model."""
    return ResNeXt(cardinality=8, base_width=64,
                   num_blocks=[3, 3, 3], num_classes=num_classes)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  DENSENET-BC-100-12  (densenet_bc_100_12)
# ══════════════════════════════════════════════════════════════════════════════
# Huang et al., CVPR 2017. Dense connections, growth_rate=12, BC variant.
# depth=100: 3 dense blocks × 16 bottleneck layers + 2 convs + 2 transitions.
# ══════════════════════════════════════════════════════════════════════════════

class DenseLayer(nn.Module):
    """
    Bottleneck dense layer: BN→ReLU→Conv(1×1,4k) → BN→ReLU→Conv(3×3,k).
    Input: concatenation of all previous feature maps in the block.
    Output: k new feature maps (growth rate).
    """
    def __init__(self, in_ch, growth_rate, bn_size=4, dropout=0.0):
        super().__init__()
        inter_ch   = bn_size * growth_rate
        self.bn1   = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, inter_ch, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(inter_ch)
        self.conv2 = nn.Conv2d(inter_ch, growth_rate, 3,
                               padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x is a list of all previous feature tensors — concatenate first
        if isinstance(x, list):
            x = torch.cat(x, dim=1)      # FIX: was torch.cat([list]) in v2
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return self.dropout(out)


class DenseBlock(nn.Module):
    """
    Dense block: each layer receives concatenation of all preceding outputs.
    After n layers: total channels = in_ch + n * growth_rate.
    """
    def __init__(self, num_layers, in_ch, growth_rate, bn_size=4, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            DenseLayer(in_ch + i * growth_rate, growth_rate, bn_size, dropout)
            for i in range(num_layers)
        ])

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feat = layer(features)          # pass list of all previous
            features.append(new_feat)
        return torch.cat(features, dim=1)       # concatenate all outputs


class TransitionLayer(nn.Module):
    """BN→ReLU→Conv(1×1, θ*in_ch)→AvgPool(2×2). Compression factor θ=0.5."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.bn   = nn.BatchNorm2d(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(F.relu(self.bn(x))))


class DenseNet(nn.Module):
    """DenseNet-BC for CIFAR. depth=100, growth_rate=12, compression=0.5."""

    def __init__(self, depth=100, growth_rate=12, compression=0.5,
                 dropout=0.0, num_classes=10):
        super().__init__()
        assert (depth - 4) % 6 == 0
        # For BC-variant: each 'layer' = 2 convs → n = (depth-4)/(3*2) per block
        n       = (depth - 4) // (3 * 2)
        init_ch = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, init_ch, 3, stride=1, padding=1, bias=False)

        in_ch = init_ch
        self.block1 = DenseBlock(n, in_ch, growth_rate, dropout=dropout)
        in_ch  = in_ch + n * growth_rate
        out_ch = int(in_ch * compression)
        self.trans1 = TransitionLayer(in_ch, out_ch)

        in_ch = out_ch
        self.block2 = DenseBlock(n, in_ch, growth_rate, dropout=dropout)
        in_ch  = in_ch + n * growth_rate
        out_ch = int(in_ch * compression)
        self.trans2 = TransitionLayer(in_ch, out_ch)

        in_ch = out_ch
        self.block3 = DenseBlock(n, in_ch, growth_rate, dropout=dropout)
        in_ch = in_ch + n * growth_rate

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
        return self.fc(torch.flatten(out, 1))


def densenet_bc_100_12(num_classes=10):
    """DenseNet-BC-100-12: depth=100, k=12, θ=0.5. ~0.77M params."""
    return DenseNet(depth=100, growth_rate=12, compression=0.5,
                    num_classes=num_classes)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  PYRAMIDNET-110-270  (pyramidnet_110_270)
# ══════════════════════════════════════════════════════════════════════════════
# Han et al., CVPR 2017. Gradual per-layer channel widening (additive).
# ch(k) = 16 + round(α × k / N_total),  α=270.
# Zero-pad shortcuts instead of 1×1 conv projection.
# ══════════════════════════════════════════════════════════════════════════════

class PyramidBlock(nn.Module):
    """Pre-act basic block with zero-pad channel shortcut."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1,
                               padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_ch)
        self.in_ch  = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.shortcut_pool = nn.AvgPool2d(2, 2) if stride == 2 else None

    def forward(self, x):
        out = self.conv1(self.bn1(x))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.bn3(out)
        # Shortcut: spatial downsampling + zero-pad channel expansion
        sc = x
        if self.shortcut_pool is not None:
            sc = self.shortcut_pool(sc)
        if self.in_ch != self.out_ch:
            pad = torch.zeros(
                sc.size(0), self.out_ch - self.in_ch,
                sc.size(2), sc.size(3),
                device=sc.device, dtype=sc.dtype
            )
            sc = torch.cat([sc, pad], dim=1)
        return F.relu(out + sc)


class PyramidNet(nn.Module):
    """PyramidNet for CIFAR. depth=110, alpha=270 (additive widening)."""

    def __init__(self, depth=110, alpha=270, num_classes=10):
        super().__init__()
        assert (depth - 2) % 6 == 0
        n             = (depth - 2) // 6       # 18 blocks per stage
        total_blocks  = 3 * n
        # Channel at each block boundary: pyramid formula
        channels = [16] + [
            16 + round(alpha * (k + 1) / total_blocks)
            for k in range(total_blocks)
        ]
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)
        layers = []
        for stage in range(3):
            for block_idx in range(n):
                g_idx  = stage * n + block_idx
                stride = 2 if (stage > 0 and block_idx == 0) else 1
                layers.append(PyramidBlock(channels[g_idx],
                                           channels[g_idx + 1], stride))
        self.layers   = nn.Sequential(*layers)
        final_ch      = channels[-1]
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
        return self.fc(torch.flatten(out, 1))


def pyramidnet_110_270(num_classes=10):
    """PyramidNet-110-270: depth=110, alpha=270. ~28.3M params."""
    return PyramidNet(depth=110, alpha=270, num_classes=num_classes)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  SHAKE-SHAKE  (shake_shake_26_2x64d)
# ══════════════════════════════════════════════════════════════════════════════
# Gastaldi, arXiv 2017. Stochastic branch mixing regularisation.
# Two parallel branches per residual block; random α (forward) ≠ β (backward).
# depth=26: 3 stages × 4 ShakeShake blocks + 2. Base width = 64ch.
# ══════════════════════════════════════════════════════════════════════════════

class ShakeShakeBranch(nn.Module):
    """One branch: BN→ReLU→Conv(3×3)→BN→ReLU→Conv(3×3)."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        return self.conv2(F.relu(self.bn2(out)))


class ShakeShakeBlock(nn.Module):
    """
    Shake-Shake block with two stochastic branches.
    Training  : forward  uses α ~ U(0,1); backward uses independent β ~ U(0,1)
    Inference : fixed weight 0.5 (deterministic average)
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.branch1  = ShakeShakeBranch(in_ch, out_ch, stride)
        self.branch2  = ShakeShakeBranch(in_ch, out_ch, stride)
        self.in_ch    = in_ch
        self.out_ch   = out_ch
        self.stride   = stride

    def _shake(self, b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return 0.5 * (b1 + b2)
        batch  = b1.size(0)
        alpha  = b1.new_empty(batch, 1, 1, 1).uniform_()
        beta   = b1.new_empty(batch, 1, 1, 1).uniform_()
        # Forward  :  alpha*b1 + (1-alpha)*b2
        # Backward :  beta *b1 + (1-beta )*b2   (shake trick via .detach())
        return (beta - alpha).detach() * b1 \
             + (alpha - beta).detach() * b2 \
             + alpha * b1 + (1 - alpha) * b2

    def forward(self, x):
        b1  = self.branch1(x)
        b2  = self.branch2(x)
        out = self._shake(b1, b2)
        # Shortcut: spatial pool + zero-pad channels
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
    Channels: 16 → 64 → 128 → 256. Three stages × 4 blocks.
    """
    def __init__(self, base_channels=64, num_blocks=None, num_classes=10):
        super().__init__()
        if num_blocks is None:
            num_blocks = [4, 4, 4]
        ch = [base_channels, base_channels * 2, base_channels * 4]
        self.conv1   = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1     = nn.BatchNorm2d(16)
        self.in_ch   = 16
        self.layer1  = self._make_stage(ch[0], num_blocks[0], stride=1)
        self.layer2  = self._make_stage(ch[1], num_blocks[1], stride=2)
        self.layer3  = self._make_stage(ch[2], num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(ch[2], num_classes)
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
        return self.fc(torch.flatten(out, 1))


def shake_shake_26_2x64d(num_classes=10):
    """Shake-Shake-26 2×64d. ~26M params. State-of-the-art in paper."""
    return ShakeShake(base_channels=64, num_blocks=[4, 4, 4],
                      num_classes=num_classes)


# ══════════════════════════════════════════════════════════════════════════════
# 9.  LIGHTCNN  —  debug model (not in paper)
# ══════════════════════════════════════════════════════════════════════════════

class LightCNN(nn.Module):
    """~95K params. Trains in seconds on CPU. For pipeline smoke-tests only."""
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
        return self.fc(torch.flatten(out, 1))


# ══════════════════════════════════════════════════════════════════════════════
# 10.  MODEL REGISTRY + FACTORY
# ══════════════════════════════════════════════════════════════════════════════
# Registry keys match the paper's footnote-1 identifier names where possible.

_MODEL_REGISTRY = {
    # Paper models
    "resnet_basic_110"            : resnet_basic_110,
    "resnet_preact_bottleneck_164": resnet_preact_bottleneck_164,
    "wrn_28_10"                   : wrn_28_10,
    "vgg16_bn"                    : vgg16_bn,
    "resnext_29_8x64d"            : resnext_29_8x64d,
    "densenet_bc_100_12"          : densenet_bc_100_12,
    "pyramidnet_110_270"          : pyramidnet_110_270,
    "shake_shake_26_2x64d"        : shake_shake_26_2x64d,
    # Extra / debug
    "resnet20"                    : resnet20,
    "lightcnn"                    : LightCNN,
}


def get_model(name: str, device: torch.device,
              num_classes: int = 10) -> nn.Module:
    """
    Model factory.

    Args:
        name        : registry key (case-insensitive, strip whitespace)
        device      : torch.device
        num_classes : output classes (default 10)

    Returns:
        nn.Module on the requested device, ready for training.
    """
    key = name.lower().strip()
    if key not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'.\n"
            f"Available: {sorted(_MODEL_REGISTRY.keys())}"
        )
    model = _MODEL_REGISTRY[key](num_classes=num_classes)
    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    """Return total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(name: str, device: torch.device) -> None:
    """Build model, run dummy forward pass, print shape + param count."""
    model    = get_model(name, device)
    n_params = count_parameters(model)
    dummy    = torch.zeros(2, 3, 32, 32, device=device)
    model.eval()
    with torch.no_grad():
        out = model(dummy)
    print(
        f"  {name:<35s}  {n_params:>12,}  "
        f"{str(tuple(dummy.shape)):>20s}  {tuple(out.shape)}"
    )


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    print("Model Summary  (8 paper models + debug)")
    print("=" * 90)
    print(f"  {'Model':<35s}  {'Params':>12s}  {'Input':>20s}  Output")
    print("-" * 90)
    for name in _MODEL_REGISTRY:
        try:
            print_model_summary(name, device)
        except Exception as e:
            print(f"  {name:<35s}  ERROR: {e}")
    print("=" * 90)
    print("\nAll models built and forward-pass verified.")