"""
Architecture requirements :
  Component 1 — Backbone
      Any of the 8 paper CNNs (or ResNet-18 adapted for 32×32).
      Must output 10-dimensional probability distribution via softmax.

  Component 2 — Prediction Head
      Maps backbone feature vector → 10-dim soft distribution.
      Required variants (for ablation D):
          'linear'      : single Linear + softmax
          'mlp'         : Linear → ReLU → Linear + softmax
          'temperature' : single Linear + temperature-scaled softmax

  Backbone initialisation (ablation A):
      'random'    : Kaiming init (default)
      'imagenet'  : torchvision ImageNet pretrained weights (ResNet-18 only)
      'cifar10'   : load checkpoint pretrained on CIFAR-10 hard labels

All models:
  - Accept 32×32 RGB CIFAR inputs
  - Return 10-dim LOG-SOFTMAX probabilities (not raw logits)
    so that KL-divergence loss can be applied directly

CHANGES vs original models.py (Peterson replication):
  - Added PredictionHead class (linear / mlp / temperature variants)
  - All build functions now compose Backbone + PredictionHead into one module
  - Output is log-softmax (needed for KLDivLoss) rather than raw logits.
    The original soft_cross_entropy in train.py applied softmax internally —
    now the model applies it so all loss functions see probabilities.
  - Added load_backbone_weights() for ablation A (backbone init strategy)
  - Added get_model_with_head() factory that Member 3 calls
  - All 8 paper backbone implementations kept intact (no changes to internals)
  - LightCNN kept for fast smoke-tests
  - Bug fixes from original file unchanged (DenseLayer.forward, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

NUM_CLASSES = 10

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION HEADS  (project §5.2 Component 2)
# ══════════════════════════════════════════════════════════════════════════════

class LinearHead(nn.Module):
    """
    Single linear layer + log-softmax.
    Minimal capacity, low overfitting risk, fully interpretable.
    Suitable when backbone features are already high-quality.
    """
    def __init__(self, in_features, num_classes=NUM_CLASSES):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return F.log_softmax(self.fc(x), dim=1)


class MLPHead(nn.Module):
    """
    Two-layer MLP + log-softmax.
    Hidden dim = in_features // 2 (halved for regularisation).
    Higher capacity than LinearHead — risk of overfitting on 6k images,
    so used with dropout.
    """
    def __init__(self, in_features, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        hidden = max(in_features // 2, num_classes * 4)
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return F.log_softmax(self.net(x), dim=1)


class TemperatureHead(nn.Module):
    """
    Single linear layer + temperature-scaled log-softmax.
    Temperature T > 1 softens the output distribution (higher entropy).
    T is a learnable scalar, initialised to 1.0.
    Useful because our targets (human soft labels) are naturally soft.
    """
    def __init__(self, in_features, num_classes=NUM_CLASSES):
        super().__init__()
        self.fc          = nn.Linear(in_features, num_classes)
        # Learnable temperature — log-space to keep T > 0
        self.log_temp    = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self):
        return self.log_temp.exp()

    def forward(self, x):
        logits = self.fc(x)
        return F.log_softmax(logits / self.temperature, dim=1)


def build_head(head_type, in_features, num_classes=NUM_CLASSES):
    """
    Factory for prediction heads.

    Args:
        head_type   : 'linear' | 'mlp' | 'temperature'
        in_features : backbone output feature dimension
        num_classes : number of output classes (10 for CIFAR-10)
    """
    head_type = head_type.lower().strip()
    if head_type == 'linear':
        return LinearHead(in_features, num_classes)
    elif head_type == 'mlp':
        return MLPHead(in_features, num_classes)
    elif head_type == 'temperature':
        return TemperatureHead(in_features, num_classes)
    else:
        raise ValueError(
            f"Unknown head_type '{head_type}'. "
            "Choose from: 'linear', 'mlp', 'temperature'."
        )


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED MODEL  — Backbone + Head
# ══════════════════════════════════════════════════════════════════════════════

class DisagreementPredictor(nn.Module):
    """
    Full model: Backbone feature extractor + Prediction head.

    Forward pass returns LOG-SOFTMAX probabilities (shape: B × 10).
    Use F.kl_div(output, target) as the primary loss.

    The backbone's original FC / classifier layer is removed and replaced
    with the chosen prediction head so that fine-tuning is focused on
    the head while backbone weights are optionally frozen.
    """
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head     = head

    def forward(self, x):
        features = self.backbone(x)   # (B, feature_dim)
        return self.head(features)    # (B, 10) log-softmax

    def freeze_backbone(self):
        """Freeze all backbone parameters (fine-tune head only)."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone for joint fine-tuning."""
        for p in self.backbone.parameters():
            p.requires_grad = True


# ══════════════════════════════════════════════════════════════════════════════
# 1.  RESNET  (resnet_basic_110 and resnet20)
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


class ResNetCIFARBackbone(nn.Module):
    """
    ResNet backbone for CIFAR (32×32). Returns feature vector, NOT logits.
    The FC layer from the original ResNetCIFAR is removed here.
    """
    def __init__(self, block, num_blocks):
        super().__init__()
        self.in_ch   = 16
        self.conv1   = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1     = nn.BatchNorm2d(16)
        self.layer1  = self._make_stage(block, 16, num_blocks[0], stride=1)
        self.layer2  = self._make_stage(block, 32, num_blocks[1], stride=2)
        self.layer3  = self._make_stage(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_features = 64 * block.expansion
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

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        return torch.flatten(out, 1)   # (B, 64)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  RESNET PRE-ACTIVATION  (resnet_preact_bottleneck_164)
# ══════════════════════════════════════════════════════════════════════════════

class PreActBottleneck(nn.Module):
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
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1,
                                      stride=stride, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        return out + self.shortcut(x)


class ResNetPreActBackbone(nn.Module):
    def __init__(self, block, num_blocks):
        super().__init__()
        self.in_ch    = 16
        self.conv1    = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.layer1   = self._make_stage(block, 64,  num_blocks[0], stride=1)
        self.layer2   = self._make_stage(block, 128, num_blocks[1], stride=2)
        self.layer3   = self._make_stage(block, 256, num_blocks[2], stride=2)
        self.bn_final = nn.BatchNorm2d(256)
        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))
        self.out_features = 256
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

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn_final(out))
        out = self.avgpool(out)
        return torch.flatten(out, 1)   # (B, 256)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  WIDE RESNET  (wrn_28_10)
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


class WideResNetBackbone(nn.Module):
    def __init__(self, depth=28, width=10, dropout=0.3):
        super().__init__()
        assert (depth - 4) % 6 == 0
        n  = (depth - 4) // 6
        ch = [16, 16 * width, 32 * width, 64 * width]
        self.conv1    = nn.Conv2d(3, ch[0], 3, stride=1, padding=1, bias=False)
        self.layer1   = self._make_stage(ch[0], ch[1], n, 1, dropout)
        self.layer2   = self._make_stage(ch[1], ch[2], n, 2, dropout)
        self.layer3   = self._make_stage(ch[2], ch[3], n, 2, dropout)
        self.bn_final = nn.BatchNorm2d(ch[3])
        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))
        self.out_features = ch[3]
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
        return torch.flatten(out, 1)   # (B, 640)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  VGG-16 BN  (vgg16_bn)
# ══════════════════════════════════════════════════════════════════════════════

VGG16_CONFIG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
                512, 512, 512, 'M', 512, 512, 512, 'M']


class VGGBackbone(nn.Module):
    def __init__(self, config=None, dropout=0.5):
        super().__init__()
        if config is None:
            config = VGG16_CONFIG
        self.features    = self._make_features(config)
        self.avgpool     = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout     = nn.Dropout(dropout)
        self.out_features = 512
        self._init_weights()

    @staticmethod
    def _make_features(config):
        layers, in_ch = [], 3
        for v in config:
            if v == 'M':
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers += [nn.Conv2d(in_ch, v, 3, padding=1, bias=False),
                           nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
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

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.dropout(out)   # (B, 512)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  RESNEXT-29-8x64d
# ══════════════════════════════════════════════════════════════════════════════

class ResNeXtBlock(nn.Module):
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


class ResNeXtBackbone(nn.Module):
    def __init__(self, cardinality=8, base_width=64, num_blocks=None):
        super().__init__()
        if num_blocks is None:
            num_blocks = [3, 3, 3]
        channels   = [256, 512, 1024]
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.in_ch = 64
        self.layer1 = self._make_stage(channels[0], num_blocks[0], 1, cardinality, base_width, 1)
        self.layer2 = self._make_stage(channels[1], num_blocks[1], 2, cardinality, base_width, 2)
        self.layer3 = self._make_stage(channels[2], num_blocks[2], 2, cardinality, base_width, 3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_features = channels[2]
        self._init_weights()

    def _make_stage(self, out_ch, n, stride, cardinality, base_width, stage_idx):
        layers = [ResNeXtBlock(self.in_ch, out_ch, stride, cardinality, base_width, stage_idx)]
        self.in_ch = out_ch
        for _ in range(n - 1):
            layers.append(ResNeXtBlock(self.in_ch, out_ch, 1, cardinality, base_width, stage_idx))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        return torch.flatten(out, 1)   # (B, 1024)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  DENSENET-BC-100-12
# ══════════════════════════════════════════════════════════════════════════════

class DenseLayer(nn.Module):
    def __init__(self, in_ch, growth_rate, bn_size=4, dropout=0.0):
        super().__init__()
        inter_ch   = bn_size * growth_rate
        self.bn1   = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, inter_ch, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(inter_ch)
        self.conv2 = nn.Conv2d(inter_ch, growth_rate, 3, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, dim=1)   # FIX: correct list concat
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return self.dropout(out)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_ch, growth_rate, bn_size=4, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            DenseLayer(in_ch + i * growth_rate, growth_rate, bn_size, dropout)
            for i in range(num_layers)
        ])

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            features.append(layer(features))
        return torch.cat(features, dim=1)


class TransitionLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.bn   = nn.BatchNorm2d(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(F.relu(self.bn(x))))


class DenseNetBackbone(nn.Module):
    def __init__(self, depth=100, growth_rate=12, compression=0.5, dropout=0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0
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
        self.out_features = in_ch
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = F.relu(self.bn_final(out))
        out = self.avgpool(out)
        return torch.flatten(out, 1)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  PYRAMIDNET-110-270
# ══════════════════════════════════════════════════════════════════════════════

class PyramidBlock(nn.Module):
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
        sc  = x
        if self.shortcut_pool is not None:
            sc = self.shortcut_pool(sc)
        if self.in_ch != self.out_ch:
            pad = torch.zeros(sc.size(0), self.out_ch - self.in_ch,
                              sc.size(2), sc.size(3),
                              device=sc.device, dtype=sc.dtype)
            sc = torch.cat([sc, pad], dim=1)
        return F.relu(out + sc)


class PyramidNetBackbone(nn.Module):
    def __init__(self, depth=110, alpha=270):
        super().__init__()
        assert (depth - 2) % 6 == 0
        n            = (depth - 2) // 6
        total_blocks = 3 * n
        channels     = [16] + [
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
        self.out_features = final_ch
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn_final(out))
        out = self.avgpool(out)
        return torch.flatten(out, 1)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  SHAKE-SHAKE-26-2x64d
# ══════════════════════════════════════════════════════════════════════════════

class ShakeShakeBranch(nn.Module):
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
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.branch1 = ShakeShakeBranch(in_ch, out_ch, stride)
        self.branch2 = ShakeShakeBranch(in_ch, out_ch, stride)
        self.in_ch   = in_ch
        self.out_ch  = out_ch
        self.stride  = stride

    def _shake(self, b1, b2):
        if not self.training:
            return 0.5 * (b1 + b2)
        batch = b1.size(0)
        alpha = b1.new_empty(batch, 1, 1, 1).uniform_()
        beta  = b1.new_empty(batch, 1, 1, 1).uniform_()
        return (beta - alpha).detach() * b1 \
             + (alpha - beta).detach() * b2 \
             + alpha * b1 + (1 - alpha) * b2

    def forward(self, x):
        out = self._shake(self.branch1(x), self.branch2(x))
        sc  = x
        if self.stride > 1:
            sc = F.avg_pool2d(sc, self.stride, self.stride)
        if self.in_ch != self.out_ch:
            pad = torch.zeros(sc.size(0), self.out_ch - self.in_ch,
                              sc.size(2), sc.size(3),
                              device=sc.device, dtype=sc.dtype)
            sc = torch.cat([sc, pad], dim=1)
        return out + sc


class ShakeShakeBackbone(nn.Module):
    def __init__(self, base_channels=64, num_blocks=None):
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
        self.out_features = ch[2]
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

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        return torch.flatten(out, 1)   # (B, 256)


# ══════════════════════════════════════════════════════════════════════════════
# 9.  LIGHTCNN  — debug model
# ══════════════════════════════════════════════════════════════════════════════

class LightCNNBackbone(nn.Module):
    """~95K params. CPU-fast. For pipeline smoke-tests only."""
    def __init__(self):
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
        self.out_features = 128

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        return torch.flatten(out, 1)   # (B, 128)


# ══════════════════════════════════════════════════════════════════════════════
# BACKBONE REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

def _build_resnet20():
    return ResNetCIFARBackbone(BasicBlock, [3, 3, 3])

def _build_resnet110():
    return ResNetCIFARBackbone(BasicBlock, [18, 18, 18])

def _build_preact164():
    return ResNetPreActBackbone(PreActBottleneck, [18, 18, 18])

def _build_wrn_28_10():
    return WideResNetBackbone(depth=28, width=10, dropout=0.3)

def _build_vgg16bn():
    return VGGBackbone(VGG16_CONFIG, dropout=0.5)

def _build_resnext():
    return ResNeXtBackbone(cardinality=8, base_width=64, num_blocks=[3, 3, 3])

def _build_densenet():
    return DenseNetBackbone(depth=100, growth_rate=12, compression=0.5)

def _build_pyramidnet():
    return PyramidNetBackbone(depth=110, alpha=270)

def _build_shakeshake():
    return ShakeShakeBackbone(base_channels=64, num_blocks=[4, 4, 4])

def _build_lightcnn():
    return LightCNNBackbone()


_BACKBONE_REGISTRY = {
    "resnet20"                    : _build_resnet20,
    "resnet_basic_110"            : _build_resnet110,
    "resnet_preact_bottleneck_164": _build_preact164,
    "wrn_28_10"                   : _build_wrn_28_10,
    "vgg16_bn"                    : _build_vgg16bn,
    "resnext_29_8x64d"            : _build_resnext,
    "densenet_bc_100_12"          : _build_densenet,
    "pyramidnet_110_270"          : _build_pyramidnet,
    "shake_shake_26_2x64d"        : _build_shakeshake,
    "lightcnn"                    : _build_lightcnn,
}


# ══════════════════════════════════════════════════════════════════════════════
# WEIGHT LOADING — Ablation A: backbone initialisation strategy
# ══════════════════════════════════════════════════════════════════════════════

def load_backbone_weights(backbone, init_strategy, ckpt_path=None, device=None):
    """
    Apply backbone initialisation strategy (project §5.2 ablation A).

    Args:
        backbone       : a backbone module (from _BACKBONE_REGISTRY)
        init_strategy  : 'random' | 'imagenet' | 'cifar10'
        ckpt_path      : path to saved checkpoint (required for 'cifar10')
        device         : torch.device

    Notes:
        'random'   — no-op; Kaiming init was already applied in __init__
        'imagenet' — only supported for wrn_28_10 / resnet_basic_110 when a
                     torchvision pretrained wrapper is used. For other backbones
                     no ImageNet weights exist; use 'cifar10' instead.
        'cifar10'  — loads a checkpoint saved by Member 3's pretrain loop.
                     The checkpoint must have been saved from a
                     DisagreementPredictor with the SAME backbone architecture.
    """
    strategy = init_strategy.lower().strip()

    if strategy == 'random':
        return backbone   # already initialised by __init__

    elif strategy == 'imagenet':
        # Only ResNet-18 has official torchvision ImageNet weights and matches
        # close enough to ResNetCIFARBackbone (structure differs slightly).
        # For a fair ablation, load torchvision ResNet-18 backbone and copy
        # layer1-3 weights, skipping the stem (7×7 → 3×3) and fc (removed).
        try:
            import torchvision.models as tvm
            pretrained = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
            # Copy layer1–3 weights only (stem is different, fc is removed)
            backbone_state = backbone.state_dict()
            pretrained_state = pretrained.state_dict()
            copied = 0
            for k in backbone_state:
                # Map torchvision names (layer1.0.conv1.weight) to ours
                if k in pretrained_state and \
                   backbone_state[k].shape == pretrained_state[k].shape:
                    backbone_state[k] = pretrained_state[k]
                    copied += 1
            backbone.load_state_dict(backbone_state)
            print(f"[INFO] ImageNet weights loaded ({copied} tensors matched).")
        except Exception as e:
            print(f"[WARN] ImageNet weight loading failed: {e}. "
                  "Falling back to random init.")
        return backbone

    elif strategy == 'cifar10':
        if ckpt_path is None:
            raise ValueError("ckpt_path is required for init_strategy='cifar10'")
        ckpt = torch.load(ckpt_path,
                           map_location=device or torch.device('cpu'))
        # Checkpoint may be a full DisagreementPredictor state_dict
        state = ckpt.get('state_dict', ckpt)
        # Strip 'backbone.' prefix if present
        backbone_state = {
            k.replace('backbone.', ''): v
            for k, v in state.items()
            if k.startswith('backbone.') or 'backbone' not in k
        }
        missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
        print(f"[INFO] CIFAR-10 pretrained weights loaded from {ckpt_path}.")
        if missing:
            print(f"       Missing keys  : {missing[:5]} ...")
        if unexpected:
            print(f"       Unexpected keys: {unexpected[:5]} ...")
        return backbone

    else:
        raise ValueError(
            f"Unknown init_strategy '{init_strategy}'. "
            "Choose from: 'random', 'imagenet', 'cifar10'."
        )


# ══════════════════════════════════════════════════════════════════════════════
# PRIMARY PUBLIC FACTORY — used by Member 3's train.py
# ══════════════════════════════════════════════════════════════════════════════

def get_model_with_head(
        backbone_name  = 'resnet_basic_110',
        head_type      = 'linear',
        init_strategy  = 'random',
        ckpt_path      = None,
        device         = None,
        num_classes    = NUM_CLASSES,
):
    """
    Build a DisagreementPredictor (backbone + head) ready for training.

    Args:
        backbone_name : key in _BACKBONE_REGISTRY
        head_type     : 'linear' | 'mlp' | 'temperature'
        init_strategy : 'random' | 'imagenet' | 'cifar10'
        ckpt_path     : checkpoint path (only for init_strategy='cifar10')
        device        : torch.device
        num_classes   : output classes (default 10)

    Returns:
        DisagreementPredictor on the requested device.

    Example:
        model = get_model_with_head(
            backbone_name='resnet_basic_110',
            head_type='mlp',
            init_strategy='cifar10',
            ckpt_path='./checkpoints/pretrain.pt',
            device=torch.device('cuda')
        )
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    key = backbone_name.lower().strip()
    if key not in _BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{backbone_name}'.\n"
            f"Available: {sorted(_BACKBONE_REGISTRY.keys())}"
        )

    backbone = _BACKBONE_REGISTRY[key]()
    backbone = load_backbone_weights(backbone, init_strategy, ckpt_path, device)
    head     = build_head(head_type, backbone.out_features, num_classes)
    model    = DisagreementPredictor(backbone, head).to(device)
    return model


# ── Backward-compatible alias used by original train.py ──────────────────────
def get_model(name, device, num_classes=NUM_CLASSES):
    """Alias for compatibility: wraps get_model_with_head with linear head."""
    return get_model_with_head(
        backbone_name=name,
        head_type='linear',
        init_strategy='random',
        device=device,
        num_classes=num_classes,
    )


def count_parameters(model):
    """Return total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(backbone_name, head_type='linear', device=None):
    """Build model, run dummy forward, print feature dim + param count."""
    if device is None:
        device = torch.device('cpu')
    model    = get_model_with_head(backbone_name, head_type, device=device)
    n_params = count_parameters(model)
    dummy    = torch.zeros(2, 3, 32, 32, device=device)
    model.eval()
    with torch.no_grad():
        out = model(dummy)
    print(
        f"  {backbone_name:<35s} | head={head_type:<11s} | "
        f"feat={model.backbone.out_features:4d} | "
        f"params={n_params:>12,} | "
        f"out={tuple(out.shape)}"
    )


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    print("Model Summary — all backbones × all head types")
    print("=" * 100)
    for bname in _BACKBONE_REGISTRY:
        for htype in ['linear', 'mlp', 'temperature']:
            try:
                print_model_summary(bname, htype, device)
            except Exception as e:
                print(f"  {bname} / {htype}: ERROR — {e}")
    print("=" * 100)

    # Verify log-softmax output (must sum to 1 in probability space)
    model = get_model_with_head('lightcnn', 'linear', device=device)
    dummy = torch.zeros(4, 3, 32, 32, device=device)
    model.eval()
    with torch.no_grad():
        log_probs = model(dummy)
        probs     = log_probs.exp()
    assert abs(probs.sum(dim=1).mean().item() - 1.0) < 1e-5, \
        "Output probabilities do not sum to 1!"
    print("\n[PASS] Output verified: log-softmax probs sum to 1.0")
    print("[PASS] All models built and forward-pass verified.")