from .vmamba import VSSMEncoder,VSSMDecoder
import torch
from torch import nn 
from .FusionStrategy import *
import torch.nn.functional as F


class CMSMamba(nn.Module):
    def __init__(self,
                 input_channels_1=3, 
                 input_channels_2=1,
                 num_classes=1,
                 depths=[2, 2, 9, 2],
                 depths_decoder=[2, 9, 2, 2],
                 dims=[48, 96, 192, 384],
                 dims_decoder=[384,192,96,48],
                 d_state=16,
                 use_checkpoint=False,
                 ssm_ratio=2.0,
                 mlp_ratio=4.0,
                 mlp_ratio_decoder=0.0,
                 dropout=0.1,
                 ):
        
        super().__init__()

        self.num_classes = num_classes

        self.encoder1 = VSSMEncoder(in_chans=input_channels_1, 
                                    dims=dims, depths=depths, 
                                    d_state=d_state, 
                                    use_checkpoint=use_checkpoint, 
                                    ssm_ratio=ssm_ratio,
                                    mlp_ratio=mlp_ratio,
                                    drop_path_rate=0.05)
        
        self.encoder2 = VSSMEncoder(in_chans=input_channels_2, 
                                    dims=dims, 
                                    depths=depths, 
                                    d_state=d_state, 
                                    use_checkpoint=use_checkpoint, 
                                    ssm_ratio=ssm_ratio,
                                    mlp_ratio=mlp_ratio,
                                    drop_path_rate=0.05)

        self.fusion_blocks = nn.ModuleList([AttentionFusionCL(c) for c in dims]) #CMR

        self.cmr_dropouts = nn.ModuleList([nn.Dropout2d(p=dropout) for _ in dims])

        self.decoder = VSSMDecoder(num_classes=num_classes,
                                    depths_decoder=depths_decoder,
                                    dims_decoder=dims_decoder,
                                    enc_dims=dims, #dims
                                    d_state=d_state,
                                    drop_path_rate=0.05,
                                    use_checkpoint=use_checkpoint,
                                    ssm_ratio=ssm_ratio,
                                    mlp_ratio=mlp_ratio_decoder)

    @staticmethod
    def _dropout2d_bhwc(x_bhwc, drop_layer: nn.Dropout2d, mc_dropout: bool):
        """
        x_bhwc: (B,H,W,C) -> (B,C,H,W) 做 Dropout2d -> (B,H,W,C)
        mc_dropout=True 時強制啟用 dropout（用於 MC Dropout 推論）
        """
        x = x_bhwc.permute(0, 3, 1, 2).contiguous()  # BHWC -> BCHW
        if mc_dropout:
            x = F.dropout2d(x, p=drop_layer.p, training=True)
        else:
            x = drop_layer(x)  # Depends on model.train()/eval() 
        x = x.permute(0, 2, 3, 1).contiguous()  # BCHW -> BHWC
        return x 
        
    def forward(self, x1, x2,mc_dropout=False):
        out_rgb = self.encoder1(x1)  # [(B,H/4,W/4,48), (B,H/8,W/8,96), (B,H/16,W/16,192), (B,H/32,W/32,384)]
        out_x = self.encoder2(x2)  # [(B,H/4,W/4,48), (B,H/8,W/8,96), (B,H/16,W/16,192), (B,H/32,W/32,384)]



        out_fused = []
        for i in range(len(out_rgb)):
            x_fuse = self.fusion_blocks[i](out_rgb[i], out_x[i], p_type='avg', spatial_type='mean')  #CMR

            if i >= 2:
                x_fuse = self._dropout2d_bhwc(x_fuse, self.cmr_dropouts[i], mc_dropout)
            out_fused.append(x_fuse)

        logits = self.decoder(out_fused)  # (B,num_classes,H,W)
        if self.num_classes == 1: 
            return torch.sigmoid(logits)
        else: 
            return logits

    
#Test
# B,H,W = 2,256,256
# x1 = torch.randn(B,3,H,W).cuda()
# x2 = torch.randn(B,1,H,W).cuda()
# model = CMSMamba(input_channels_1=3,input_channels_2=1,num_classes=5).cuda()
# y = model(x1,x2)
# print(y.shape)  