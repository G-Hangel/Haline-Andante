import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNetExpert(nn.Module):
    """Sub-rede Especialista leve (U-Net compacta)."""
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(64, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        d2 = self.up2(e3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)

class GatingNetwork(nn.Module):
    """Rede Roteadora: Avalia a entrada e decide o peso de cada Expert (2 peritos)."""
    def __init__(self, num_experts=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1), # 128x128
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1), # 64x64
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), # 1x1
            nn.Flatten()
        )
        self.fc = nn.Linear(32, num_experts)

    def forward(self, x):
        feat = self.features(x)
        logits = self.fc(feat)
        weights = F.softmax(logits, dim=1) # Pesos normalizados (soma = 1.0)
        return weights

class MoEGeometricNet(nn.Module):
    """Modelo completo MoE: Junta os 2 Experts + Gating Network."""
    def __init__(self):
        super().__init__()
        self.expert_cleaning = UNetExpert()     # Expert 1: Limpeza de traço
        self.expert_regularizer = UNetExpert()  # Expert 2: Regularizador geométrico
        self.gating = GatingNetwork(num_experts=2)

    def forward(self, x):
        # Saída dos especialistas (logits)
        out_clean = self.expert_cleaning(x)
        out_reg = self.expert_regularizer(x)

        # Pesos do roteador [Batch, 2]
        weights = self.gating(x)

        # Combinação ponderada das saídas
        w1 = weights[:, 0].view(-1, 1, 1, 1)
        w2 = weights[:, 1].view(-1, 1, 1, 1)

        combined_logits = (w1 * out_clean) + (w2 * out_reg)
        return combined_logits, weights, out_clean, out_reg