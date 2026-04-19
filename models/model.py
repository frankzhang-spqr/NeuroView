"""
Segmentation model definitions.
"""
import torch
import torch.nn as nn


def infer_segmentation_variant(state_dict):
    if any(key.startswith("stem.") or key.startswith("decoders.") for key in state_dict.keys()):
        return "enhanced"
    return "legacy"


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels),
    )


class UNet(nn.Module):
    def __init__(self, in_channels=4, n_class=1):
        super().__init__()
        self.e1 = conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.e2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.e3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.e4 = conv_block(128, 256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.b = conv_block(256, 512)
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d1 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d2 = conv_block(256, 128)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d3 = conv_block(128, 64)
        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.d4 = conv_block(64, 32)
        self.outconv = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        s1 = self.e1(x)
        p1 = self.pool1(s1)
        s2 = self.e2(p1)
        p2 = self.pool2(s2)
        s3 = self.e3(p2)
        p3 = self.pool3(s3)
        s4 = self.e4(p3)
        p4 = self.pool4(s4)
        b = self.b(p4)
        d1 = self.d1(torch.cat([self.upconv1(b), s4], dim=1))
        d2 = self.d2(torch.cat([self.upconv2(d1), s3], dim=1))
        d3 = self.d3(torch.cat([self.upconv3(d2), s2], dim=1))
        d4 = self.d4(torch.cat([self.upconv4(d3), s1], dim=1))
        return self.outconv(d4)


class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            SqueezeExcite(out_channels),
        )
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.block(x) + self.skip(x))


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = nn.Sequential(
            ResidualBlock(out_channels + skip_channels, out_channels),
            ResidualBlock(out_channels, out_channels),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class EnhancedUNet(nn.Module):
    def __init__(self, in_channels=4, n_class=1, widths=(32, 64, 128, 256, 384)):
        super().__init__()
        self.stem = ResidualBlock(in_channels, widths[0])
        self.encoders = nn.ModuleList([
            ResidualBlock(widths[0], widths[1], stride=2),
            ResidualBlock(widths[1], widths[2], stride=2),
            ResidualBlock(widths[2], widths[3], stride=2),
            ResidualBlock(widths[3], widths[4], stride=2),
        ])
        self.bridge = nn.Sequential(
            ResidualBlock(widths[4], widths[4]),
            ResidualBlock(widths[4], widths[4]),
        )
        self.decoders = nn.ModuleList([
            DecoderBlock(widths[4], widths[3], widths[3]),
            DecoderBlock(widths[3], widths[2], widths[2]),
            DecoderBlock(widths[2], widths[1], widths[1]),
            DecoderBlock(widths[1], widths[0], widths[0]),
        ])
        self.head = nn.Sequential(
            nn.Conv2d(widths[0], widths[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(widths[0], n_class, kernel_size=1),
        )

    def forward(self, x):
        skips = [self.stem(x)]
        current = skips[0]
        for encoder in self.encoders:
            current = encoder(current)
            skips.append(current)

        current = self.bridge(current)
        decoder_skips = skips[:-1][::-1]
        for decoder, skip in zip(self.decoders, decoder_skips):
            current = decoder(current, skip)
        return self.head(current)


def build_model(in_channels=4, n_class=1, variant="enhanced"):
    if variant == "legacy":
        return UNet(in_channels, n_class)
    if variant == "enhanced":
        return EnhancedUNet(in_channels, n_class)
    raise ValueError(f"Unsupported segmentation variant: {variant}")
