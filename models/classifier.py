"""
Classification model definitions.
"""
import torch.nn as nn


def infer_classifier_variant(state_dict):
    if any(key.startswith("stem.") or key.startswith("stages.") for key in state_dict.keys()):
        return "enhanced"
    return "legacy"


class TumorClassifier(nn.Module):
    def __init__(self, in_channels=4, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 30 * 30, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class ClassifierBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            SEBlock(out_channels),
        )
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.main(x) + self.skip(x))


class EnhancedTumorClassifier(nn.Module):
    def __init__(self, in_channels=4, num_classes=2, widths=(32, 64, 128, 256)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, widths[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.SiLU(inplace=True),
        )
        self.stages = nn.Sequential(
            ClassifierBlock(widths[0], widths[0]),
            ClassifierBlock(widths[0], widths[1], stride=2),
            ClassifierBlock(widths[1], widths[1]),
            ClassifierBlock(widths[1], widths[2], stride=2),
            ClassifierBlock(widths[2], widths[2]),
            ClassifierBlock(widths[2], widths[3], stride=2),
            ClassifierBlock(widths[3], widths[3]),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(widths[3], 192),
            nn.SiLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x)
        return self.classifier(x)


def build_classifier(in_channels=4, num_classes=2, variant="enhanced"):
    if variant == "legacy":
        return TumorClassifier(in_channels=in_channels, num_classes=num_classes)
    if variant == "enhanced":
        return EnhancedTumorClassifier(in_channels=in_channels, num_classes=num_classes)
    raise ValueError(f"Unsupported classifier variant: {variant}")
