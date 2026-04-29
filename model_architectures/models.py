"""
model architectures/models.py
=======================
Member 2 — Model Architectures
Provides all CNN architectures used in Peterson et al. (ICCV 2019):
  • ResNet family  : ResNet-110 (deep), ResNet-20 (lightweight)
  • Wide ResNet    : WRN-28-10
  • VGG            : VGG-16 with BatchNorm
  • LightCNN       : tiny debug model
  • Factory        : get_model(name, device) — single entry-point for all models

All models:
  - Accept 32×32 RGB inputs  (CIFAR-10 / CIFAR-10H)
  - Output 10-class logits   (no softmax — use F.softmax or F.log_softmax outside)
  - Use AdaptiveAvgPool so spatial size never needs hard-coding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# 1.  RESNET FAMILY
# ══════════════════════════════════════════════════════════════════════════════
# Architecture: He et al., "Deep Residual Learning for Image Recognition",
#               CVPR 2016.
#
# Key idea: skip connections let gradients flow unimpeded through very deep
# networks, solving the vanishing-gradient problem.
#
# For CIFAR (32×32) the paper uses a simplified design:
#   • First conv:  3×3, 16 filters, stride 1  (no max-pool, image stays 32×32)
#   • 3 stages of n BasicBlocks each, doubling channels and halving spatial
#     size at each stage transition via stride-2 convolution.
#   • Global average pooling → FC(10)
#
# Depth formula: depth = 6n + 2
#   n=3  → depth=20  (ResNet-20,  lightweight, fast training)
#   n=18 → depth=110 (ResNet-110, deeper, better accuracy)
# ══════════════════════════════════════════════════════════════════════════════

class BasicBlock(nn.Module):
    """
    The fundamental building block of ResNet for CIFAR.

    Structure:
        x ──► Conv─BN─ReLU ──► Conv─BN ──(+)──► ReLU ──► out
        │                                  ▲
        └──────────── shortcut ────────────┘

    The shortcut is either:
      • an identity (when input and output channels / strides match), or
      • a 1×1 conv that adjusts channels and spatial size.

    Parameters
    ----------
    in_channels  : number of input feature maps
    out_channels : number of output feature maps
    stride       : stride of the first conv (1 = same size, 2 = halve spatial)
    """
    expansion = 1   # BasicBlock keeps channels the same across the block

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        # ── Main path ────────────────────────────────────────────────────────
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # ── Shortcut (skip connection) ────────────────────────────────────────
        # Needed whenever the shortcut tensor and the main-path tensor differ
        # in either spatial size (stride > 1) or channel count.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))   # conv → BN → ReLU
        out = self.bn2(self.conv2(out))          # conv → BN  (no ReLU yet)
        out = out + self.shortcut(x)             # residual addition
        out = F.relu(out)                        # ReLU after addition
        return out


class ResNetCIFAR(nn.Module):
    """
    General ResNet for CIFAR-10 (32×32 inputs).

    Parameters
    ----------
    block      : block class to use (BasicBlock)
    num_blocks : list of 3 ints — number of blocks per stage
                 e.g. [3,3,3] for ResNet-20, [18,18,18] for ResNet-110
    num_classes: output classes (10 for CIFAR-10)
    """

    def __init__(self, block, num_blocks: list, num_classes: int = 10):
        super().__init__()
        self.in_channels = 16   # channels after the very first conv

        # ── Stem: one 3×3 conv, keeps spatial size 32×32 ─────────────────────
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)

        # ── Three residual stages ─────────────────────────────────────────────
        # Stage 1: 16 ch, spatial 32×32
        self.layer1 = self._make_stage(block, 16,  num_blocks[0], stride=1)
        # Stage 2: 32 ch, spatial 16×16  (stride-2 in first block)
        self.layer2 = self._make_stage(block, 32,  num_blocks[1], stride=2)
        # Stage 3: 64 ch, spatial  8×8   (stride-2 in first block)
        self.layer3 = self._make_stage(block, 64,  num_blocks[2], stride=2)

        # ── Head ──────────────────────────────────────────────────────────────
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))   # → (B, 64, 1, 1)
        self.fc      = nn.Linear(64 * block.expansion, num_classes)

        # ── Weight initialisation ─────────────────────────────────────────────
        self._init_weights()

    def _make_stage(self, block, out_channels: int,
                    num_blocks: int, stride: int) -> nn.Sequential:
        """Build one residual stage (a sequence of blocks)."""
        # Only the first block uses the given stride (to downsample).
        # All subsequent blocks use stride=1.
        strides = [stride] + [1] * (num_blocks - 1)
        layers  = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def _init_weights(self):
        """Kaiming (He) initialisation for conv layers; zero-init BN bias."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))   # stem
        out = self.layer1(out)                   # stage 1
        out = self.layer2(out)                   # stage 2
        out = self.layer3(out)                   # stage 3
        out = self.avgpool(out)                  # global avg pool
        out = torch.flatten(out, 1)              # (B, 64)
        out = self.fc(out)                       # logits (B, 10)
        return out


def resnet20(num_classes: int = 10) -> ResNetCIFAR:
    """
    ResNet-20: depth = 6×3 + 2 = 20 layers.
    Lightweight; good for quick experiments or limited compute.
    ~0.27 M parameters.
    """
    return ResNetCIFAR(BasicBlock, [3, 3, 3], num_classes)


def resnet110(num_classes: int = 10) -> ResNetCIFAR:
    """
    ResNet-110: depth = 6×18 + 2 = 110 layers.
    Deeper variant used in the original paper for best accuracy.
    ~1.7 M parameters.
    """
    return ResNetCIFAR(BasicBlock, [18, 18, 18], num_classes)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  WIDE RESNET  (WRN-28-10)
# ══════════════════════════════════════════════════════════════════════════════
# Architecture: Zagoruyko & Komodakis, "Wide Residual Networks",
#               BMVC 2016.
#
# Observation: making networks *wider* (more filters per layer) is often more
# efficient than making them deeper.
#
# Key parameters:
#   depth k : total number of conv layers (28 here)
#   width k : multiplicative factor applied to baseline channel counts
#             [16, 32, 64] → [16k, 32k, 64k]  (k=10 here, so 160/320/640)
#
# Each residual block uses a Pre-activation design with Dropout inside:
#   BN─ReLU─Conv─Dropout─BN─ReLU─Conv  (+ shortcut)
# ══════════════════════════════════════════════════════════════════════════════

class WideBasicBlock(nn.Module):
    """
    Wide ResNet basic block with pre-activation and optional dropout.

    Pre-activation (BN→ReLU before conv) is known to improve training
    stability in very wide networks.

    Parameters
    ----------
    in_channels  : input channels
    out_channels : output channels (wider than standard ResNet)
    stride       : spatial stride for the first conv
    dropout_rate : dropout probability inserted between the two convs
    """
    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, dropout_rate: float = 0.0):
        super().__init__()

        self.bn1   = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.bn2   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)

        # Shortcut — adjust dimensions if needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-activation path
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = out + self.shortcut(x)
        return out


class WideResNet(nn.Module):
    """
    Wide ResNet — WRN-depth-width.
    Default: WRN-28-10  (depth=28, width=10).

    Parameters
    ----------
    depth        : total layers; must satisfy (depth-4) % 6 == 0
    width        : widening factor (k) applied to [16,32,64]
    dropout_rate : dropout in each block (paper uses 0.3 for WRN-28-10)
    num_classes  : output classes
    """
    def __init__(self, depth: int = 28, width: int = 10,
                 dropout_rate: float = 0.3, num_classes: int = 10):
        super().__init__()

        assert (depth - 4) % 6 == 0, \
            "WideResNet depth must satisfy (depth-4) % 6 == 0"
        n = (depth - 4) // 6   # blocks per stage

        # Channel counts: widened by factor k
        channels = [16, 16 * width, 32 * width, 64 * width]

        # ── Stem ──────────────────────────────────────────────────────────────
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3,
                               stride=1, padding=1, bias=False)

        # ── Three wide stages ─────────────────────────────────────────────────
        self.layer1 = self._make_stage(channels[0], channels[1],
                                       n, stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_stage(channels[1], channels[2],
                                       n, stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_stage(channels[2], channels[3],
                                       n, stride=2, dropout_rate=dropout_rate)

        # ── Head ──────────────────────────────────────────────────────────────
        self.bn_final = nn.BatchNorm2d(channels[3])
        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))
        self.fc       = nn.Linear(channels[3], num_classes)

        self._init_weights()

    @staticmethod
    def _make_stage(in_ch: int, out_ch: int, n: int,
                    stride: int, dropout_rate: float) -> nn.Sequential:
        layers = [WideBasicBlock(in_ch, out_ch, stride=stride,
                                 dropout_rate=dropout_rate)]
        for _ in range(n - 1):
            layers.append(WideBasicBlock(out_ch, out_ch, stride=1,
                                         dropout_rate=dropout_rate))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn_final(out))   # final BN+ReLU (pre-act style)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def wrn_28_10(num_classes: int = 10) -> WideResNet:
    """
    WRN-28-10: depth=28, width=10, dropout=0.3.
    ~36.5 M parameters. The best-performing model in the paper.
    """
    return WideResNet(depth=28, width=10, dropout_rate=0.3,
                      num_classes=num_classes)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  VGG-16 WITH BATCH NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════
# Architecture: Simonyan & Zisserman, "Very Deep Convolutional Networks
#               for Large-Scale Image Recognition", ICLR 2015.
#
# Original VGG was designed for 224×224 ImageNet images.  We adapt it for
# CIFAR's 32×32 by:
#   • Keeping all conv blocks (same filter pattern).
#   • Using AdaptiveAvgPool(1,1) in place of the large FC layers so the
#     spatial bottleneck is handled automatically regardless of input size.
#   • Adding BatchNorm after every conv (VGG-16-BN variant).
#
# VGG-16 configuration (block → channels, repeats):
#   [64,64] → pool → [128,128] → pool → [256,256,256] → pool
#   → [512,512,512] → pool → [512,512,512] → pool → GAP → FC(10)
# ══════════════════════════════════════════════════════════════════════════════

# VGG-16 channel layout: list of ints = conv filters, 'M' = max-pool
VGG16_CONFIG = [
    64, 64, 'M',
    128, 128, 'M',
    256, 256, 256, 'M',
    512, 512, 512, 'M',
    512, 512, 512, 'M'
]


class VGG(nn.Module):
    """
    VGG-16 with Batch Normalisation, adapted for CIFAR-32×32.

    Parameters
    ----------
    config      : list describing conv filter counts and 'M' for max-pool
    num_classes : output classes
    dropout     : dropout rate in the classifier head
    """
    def __init__(self, config: list = VGG16_CONFIG,
                 num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        self.features    = self._make_features(config)
        self.avgpool     = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier  = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        self._init_weights()

    @staticmethod
    def _make_features(config: list) -> nn.Sequential:
        """Build the convolutional feature extractor from the config list."""
        layers: list = []
        in_ch = 3
        for v in config:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers += [
                    nn.Conv2d(in_ch, v, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True),
                ]
                in_ch = v
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)       # conv blocks
        out = self.avgpool(out)      # (B, 512, 1, 1)
        out = torch.flatten(out, 1)  # (B, 512)
        out = self.classifier(out)   # (B, 10)
        return out


def vgg16_bn(num_classes: int = 10) -> VGG:
    """VGG-16 with BatchNorm. ~14.7 M parameters."""
    return VGG(VGG16_CONFIG, num_classes=num_classes)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  LIGHTCNN  (fast debug model)
# ══════════════════════════════════════════════════════════════════════════════
# A tiny 4-layer CNN. NOT used for real experiments — only for quickly
# verifying that the data pipeline, training loop, and evaluation code all
# work before committing to a multi-hour training run.
#
# Architecture:
#   Conv(32)→BN→ReLU→Pool → Conv(64)→BN→ReLU→Pool → GAP → FC(10)
# ══════════════════════════════════════════════════════════════════════════════

class LightCNN(nn.Module):
    """
    Lightweight debug CNN for CIFAR-10 (32×32 input).
    ~200 K parameters; trains in seconds on CPU.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),           # 32 → 16

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),           # 16 → 8

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),           # 8 → 4
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))   # 4 → 1
        self.fc      = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  FACTORY  — get_model(name, device)
# ══════════════════════════════════════════════════════════════════════════════
# Single entry-point for Member 3 (training) and Member 4 (evaluation).
# Usage:
#   model = get_model("resnet20", device)
#   model = get_model("wrn28_10", device)
# ══════════════════════════════════════════════════════════════════════════════

# Registry: string name → constructor function
_MODEL_REGISTRY: dict = {
    "resnet20"  : resnet20,
    "resnet110" : resnet110,
    "wrn28_10"  : wrn_28_10,
    "vgg16"     : vgg16_bn,
    "lightcnn"  : LightCNN,
}


def get_model(name: str, device: torch.device,
              num_classes: int = 10) -> nn.Module:
    """
    Model factory.  Returns the requested architecture moved to `device`.

    Parameters
    ----------
    name        : one of  'resnet20', 'resnet110', 'wrn28_10',
                          'vgg16', 'lightcnn'
    device      : torch.device('cpu') or torch.device('cuda')
    num_classes : number of output logits (default 10 for CIFAR-10)

    Returns
    -------
    nn.Module on the requested device, ready for training.

    Raises
    ------
    ValueError if name is not in the registry.
    """
    name = name.lower().strip()
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )
    model = _MODEL_REGISTRY[name](num_classes=num_classes)
    model = model.to(device)
    return model


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(name: str, device: torch.device) -> None:
    """
    Instantiate a model, print its parameter count, and run a single
    forward pass with a dummy batch to confirm shapes are correct.
    """
    model = get_model(name, device)
    n_params = count_parameters(model)
    dummy = torch.zeros(2, 3, 32, 32, device=device)   # batch of 2
    with torch.no_grad():
        out = model(dummy)
    print(f"[{name:12s}]  params={n_params:>10,d}   "
          f"input={tuple(dummy.shape)}  output={tuple(out.shape)}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  QUICK SELF-TEST  (run with:  python models.py)
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    print("Model Summary")
    print("=" * 65)
    for name in _MODEL_REGISTRY:
        print_model_summary(name, device)
    print("=" * 65)
    print("\nAll models built and forward-pass verified successfully.")