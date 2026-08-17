import torch
import torch.nn as nn
import torch.nn.functional as F


EPSILON = 1e-5


#### Channel last
class AttentionFusionCL(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.proj = nn.Linear(in_channels * 2, in_channels, bias=True)

    def forward(self, t1, t2, p_type='avg', spatial_type='mean'):
        # t1, t2: (B,H,W,C)
        f_ch = self.channel_fusion(t1, t2, p_type)        # (B,H,W,C)
        f_sp = self.spatial_fusion(t1, t2, spatial_type)  # (B,H,W,C)
        cat = torch.cat([f_ch, f_sp], dim=-1)             # (B,H,W,2C)

        out = self.proj(cat)                              # (B,H,W,C)
        return out

    # ---- attention submodule（channel-last）----
    def channel_fusion(self, t1, t2, p_type='avg'):
        # GlobalPool H,W, get (B,1,1,C)
        if p_type in ['avg', 'attention_avg']:
            gp1 = t1.mean(dim=(1,2), keepdim=True)
            gp2 = t2.mean(dim=(1,2), keepdim=True)
        else:  # 'max'
            gp1, _ = t1.amax(dim=(1,2), keepdim=True), None
            gp2, _ = t2.amax(dim=(1,2), keepdim=True), None

        w = torch.softmax(torch.cat([gp1, gp2], dim=-1), dim=-1)  # (B,1,1,2C)
        w1, w2 = w[..., :gp1.size(-1)], w[..., gp1.size(-1):]     # (B,1,1,C)
        return w1 * t1 + w2 * t2                                  # broadcast -> (B,H,W,C)

    def spatial_fusion(self, t1, t2, spatial_type='mean'):
        # GlobalPool C, get (B,H,W,1)
        if spatial_type == 'mean':
            s1 = t1.mean(dim=-1, keepdim=True)   # (B,H,W,1)
            s2 = t2.mean(dim=-1, keepdim=True)
        elif spatial_type == 'sum':
            s1 = t1.sum(dim=-1, keepdim=True)
            s2 = t2.sum(dim=-1, keepdim=True)

        w = torch.softmax(torch.cat([s1, s2], dim=-1), dim=-1)  # (B,H,W,2)
        w1, w2 = w[..., :1], w[..., 1:]                         # (B,H,W,1)
        return w1 * t1 + w2 * t2                                # (B,H,W,C)
