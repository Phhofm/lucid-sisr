"""ICNet architecture for image complexity scoring.

Original source: https://github.com/umzi2/Dataset_Preprocessing
Adapted for LUCID with modern torchvision API.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class slam(nn.Module):
    """Spatial Linear Attention Module - applies attention based on spatial pooling."""

    def __init__(self, spatial_dim: int):
        super(slam, self).__init__()
        self.spatial_dim = spatial_dim
        self.linear = nn.Sequential(
            nn.Linear(spatial_dim**2, 512), nn.ReLU(), nn.Linear(512, 1), nn.Sigmoid()
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        n, c, h, w = feature.shape
        if h != self.spatial_dim:
            x = F.interpolate(
                feature,
                size=(self.spatial_dim, self.spatial_dim),
                mode="bilinear",
                align_corners=True,
            )
        else:
            x = feature

        x = x.view(n, c, -1)
        x = self.linear(x)
        x = x.unsqueeze(dim=3)
        out = x.expand_as(feature) * feature
        return out


class to_map(nn.Module):
    """Maps feature channels to a single complexity map."""

    def __init__(self, channels: int):
        super(to_map, self).__init__()
        self.to_map = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.to_map(feature)


class conv_bn_relu(nn.Module):
    """Convolution followed by BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1, stride: int = 1):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class up_conv_bn_relu(nn.Module):
    """Upsampling followed by convolution, BatchNorm, and ReLU."""

    def __init__(
        self,
        scale: int,
        in_channels: int,
        out_channels: int = 64,
        kernal_size: int = 1,
        padding: int = 0,
        stride: int = 1,
    ):
        super(up_conv_bn_relu, self).__init__()
        self.upSample = (
            nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True)
            if scale != 1
            else nn.Identity()
        )
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernal_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upSample(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ICNet(nn.Module):
    """Image Complexity Network for scoring image tiles.

    Uses two ResNet-18 branches (detail and context) to predict per-pixel
    complexity maps and overall image complexity scores.
    """

    def __init__(self, is_pretrain: bool = True, size1: int = 512, size2: int = 256):
        super(ICNet, self).__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if is_pretrain else None
        resnet18Pretrained1 = resnet18(weights=weights)
        resnet18Pretrained2 = resnet18(weights=weights)

        self.size1 = size1
        self.size2 = size2

        # Detail branch
        self.b1_1 = nn.Sequential(*list(resnet18Pretrained1.children())[:5])
        self.b1_1_slam = slam(32)

        self.b1_2 = list(resnet18Pretrained1.children())[5]
        self.b1_2_slam = slam(32)

        # Context branch
        self.b2_1 = nn.Sequential(*list(resnet18Pretrained2.children())[:5])
        self.b2_1_slam = slam(32)

        self.b2_2 = list(resnet18Pretrained2.children())[5]
        self.b2_2_slam = slam(32)

        self.b2_3 = list(resnet18Pretrained2.children())[6]
        self.b2_3_slam = slam(16)

        self.b2_4 = list(resnet18Pretrained2.children())[7]
        self.b2_4_slam = slam(8)

        # Upsampling layers — scale chosen so detail (b1_2, 1/8 input) and
        # context (b2_4, 1/64 of x2 = 1/128 of input) produce matching sizes.
        # For 512 input: detail=64×64, context=8×8 → up2 needs 8× upsampling.
        self.up1 = up_conv_bn_relu(scale=1, in_channels=128, out_channels=256)
        self.up2 = up_conv_bn_relu(scale=8, in_channels=512, out_channels=256)

        # Map prediction head
        self.to_map_f = conv_bn_relu(256 * 2, 256 * 2)
        self.to_map_f_slam = slam(32)
        self.to_map = to_map(256 * 2)

        # Score prediction head
        self.to_score_f = conv_bn_relu(256 * 2, 256 * 2)
        self.to_score_f_slam = slam(32)
        self.head = nn.Sequential(nn.Linear(256 * 2, 512), nn.ReLU(), nn.Linear(512, 1), nn.Sigmoid())
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (complexity_score, complexity_map).

        The architecture is size-flexible: SLAM attention modules interpolate
        to a fixed ``spatial_dim`` internally, the up_conv uses a multiplicative
        ``scale_factor``, and the head uses ``AdaptiveAvgPool2d((1, 1))``.
        The constructor ``size1`` parameter is kept so weights and BatchNorm
        running statistics are interpreted consistently, but the model runs
        forward on any square input.

        The score drift introduced by non-512 input is corrected at the
        pipeline layer by ``lucid.resources.icnet_calibration.apply_calibration``.
        """
        if x1.shape[2] != x1.shape[3]:
            raise ValueError(
                f"ICNet expects square input, got shape {tuple(x1.shape)}"
            )
        x2 = F.interpolate(x1, size=(self.size2, self.size2), mode="bilinear", align_corners=True)

        x1_out = self.b1_2_slam(self.b1_2(self.b1_1_slam(self.b1_1(x1))))
        x2_out = self.b2_2_slam(self.b2_2(self.b2_1_slam(self.b2_1(x2))))
        x2_out = self.b2_4_slam(self.b2_4(self.b2_3_slam(self.b2_3(x2_out))))

        x1_out = self.up1(x1_out)
        x2_out = self.up2(x2_out)
        x_cat = torch.cat((x1_out, x2_out), dim=1)

        cly_map = self.to_map(self.to_map_f_slam(self.to_map_f(x_cat)))

        score_feature = self.to_score_f_slam(self.to_score_f(x_cat))
        score_feature = self.avgpool(score_feature)
        score_feature = score_feature.squeeze()
        score = self.head(score_feature)
        score = score.squeeze()

        return score, cly_map
