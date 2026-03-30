import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

from .config_3d import TABULAR_COLS


class VideoEncoder3D(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        base = r3d_18(weights=weights)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.out_dim = base.fc.in_features  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  
        feats = self.backbone(x)      
        feats = feats.view(B, -1)     
        return feats


class DeepPitchModel3D(nn.Module):
    def __init__(self, hidden: int = 256, pretrained_3d: bool = True):
        super().__init__()
        self.video_encoder = VideoEncoder3D(pretrained=pretrained_3d)
        vid_dim = self.video_encoder.out_dim
        tab_dim = len(TABULAR_COLS)

        self.tab_mlp = nn.Sequential(
            nn.Linear(tab_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.shared = nn.Sequential(
            nn.Linear(vid_dim + 64, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.head_class = nn.Linear(128, 1)    
        self.head_zone  = nn.Linear(128, 14)   
        self.head_px    = nn.Linear(128, 1)    
        self.head_pz    = nn.Linear(128, 1)    

    def forward(self, video, tabular):
        v = self.video_encoder(video)
        t = self.tab_mlp(tabular)
        x = torch.cat([v, t], dim=1)
        h = self.shared(x)

        logit_class = self.head_class(h).squeeze(1)
        logits_zone = self.head_zone(h)
        px = self.head_px(h).squeeze(1)
        pz = self.head_pz(h).squeeze(1)

        return logit_class, logits_zone, px, pz
