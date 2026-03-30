# src/models.py
import torch
import torch.nn as nn
import torchvision.models as models

from .config import TABULAR_COLS


class VideoEncoder(nn.Module):
    """
    Frame-wise ResNet18 → temporal GRU → final hidden state.
    Output: [B, 2 * hidden_temporal]
    """
    def __init__(self, pretrained: bool = True, hidden_temporal: int = 256):
        super().__init__()

        base = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        modules = list(base.children())[:-1]  # drop final FC
        self.cnn = nn.Sequential(*modules)    # [B,512,1,1]
        self.feat_dim = 512

        self.gru = nn.GRU(
            input_size=self.feat_dim,
            hidden_size=hidden_temporal,
            batch_first=True,
            bidirectional=True,
        )
        self.out_dim = hidden_temporal * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,T,C,H,W]
        returns: [B, 2 * hidden_temporal]
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.cnn(x)                 # [B*T, 512, 1, 1]
        feats = feats.view(B, T, -1)        # [B, T, 512]

        # GRU over time
        _, h = self.gru(feats)              # h: [2, B, hidden]
        h = h.transpose(0, 1).contiguous()  # [B, 2, hidden]
        h = h.view(B, -1)                   # [B, 2*hidden]
        return h


class DeepPitchModel(nn.Module):
    """
    Video (ResNet18+GRU) + tabular MLP → shared trunk → 4 heads:
      - pitch_class (strike vs ball, BCE with logits)
      - zone (14-way CE)
      - plate_x, plate_z (L1)
    """
    def __init__(self, tab_dim: int, hidden: int = 256, temporal_hidden: int = 256):
        super().__init__()

        self.video_encoder = VideoEncoder(
            pretrained=True,
            hidden_temporal=temporal_hidden,
        )

        self.tab_mlp = nn.Sequential(
            nn.Linear(tab_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        shared_in_dim = self.video_encoder.out_dim + 64

        self.shared = nn.Sequential(
            nn.Linear(shared_in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # heads
        self.head_class = nn.Linear(128, 1)   # strike vs ball
        self.head_zone  = nn.Linear(128, 14)  # zones 1..14 (0..13 indexed)
        self.head_px    = nn.Linear(128, 1)   # plate_x
        self.head_pz    = nn.Linear(128, 1)   # plate_z

    def forward(self, video: torch.Tensor, tabular: torch.Tensor):
        """
        video:  [B,T,C,H,W]
        tabular:[B, F]
        """
        v = self.video_encoder(video)
        t = self.tab_mlp(tabular)
        x = torch.cat([v, t], dim=1)
        h = self.shared(x)

        logit_class = self.head_class(h).squeeze(1)  # [B]
        logits_zone = self.head_zone(h)              # [B,14]
        px          = self.head_px(h).squeeze(1)     # [B]
        pz          = self.head_pz(h).squeeze(1)     # [B]

        return logit_class, logits_zone, px, pz
