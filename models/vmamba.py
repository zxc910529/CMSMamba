import math
from functools import partial
from typing import  Callable
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

# fvcore flops =======================================
def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try: 
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)


def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    assert not with_complex 
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L    
    return flops

# this is only for selective_scan_ref...
def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    import numpy as np
    
    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
    

    assert not with_complex

    flops = 0 # below code flops = 0

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
  
    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L  
    return flops

# Using 4*4 Conv to get patch (b*w*h*c)
class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x

#Reduce 2 times, for encoder
class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H//2, W//2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x) # LN
        x = self.reduction(x) #4*C -> 2*C

        return x
    
#Expand 2 times, for decoder
class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim*2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale) #(B,H,W,C) -> (B,H*2,W*2,C/2)
        x= self.norm(x)

        return x
    
# Expand 4 times, for final output
class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim  
        self.dim_scale = dim_scale 
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x= self.norm(x)

        return x
    

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def get_diagonal_indices(H, W, direction='tlbr'):
    """
    Generate the pixel index sequence for diagonal scanning.
    direction: 
        'tlbr' = Top-left to Bottom-right diagonal (\)
        'trbl' = Top-right to Bottom-left diagonal (/)
    Returns: indices (L,) for gather/scatter
    """
    indices = []
    if direction == 'tlbr':  # diagonal (\)
        for d in range(-(H-1), W):
            for i in range(H):
                j = i + d
                if 0 <= j < W:
                    indices.append(i * W + j)
    elif direction == 'trbl':  # diagonal (/)
        for d in range(H + W - 1):
            for i in range(H):
                j = d - i
                if 0 <= j < W:
                    indices.append(i * W + j)
    return torch.tensor(indices, dtype=torch.long)

class SS2D(nn.Module):
    def __init__(
        self,
        d_model,   # 96
        d_state=8,
        d_conv=3,
        ssm_ratio=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
    ):
        super().__init__()
        self.d_model = d_model  
        self.d_state = d_state  
        self.d_conv = d_conv   
        self.expand = ssm_ratio   
        self.d_inner = int(self.expand * self.d_model)  
        self.dt_rank = math.ceil(self.d_model / 16)  

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,  
            out_channels=self.d_inner,  
            kernel_size=d_conv,  
            padding=(d_conv - 1) // 2,   
            bias=conv_bias,  
            groups=self.d_inner,  
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
        )
        #Initialize x_proj weights as a single parameter
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=8, N, inner)
        del self.x_proj

        #Initialize dt_proj weights as a single parameter
        self.dt_projs = tuple(
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor) for _ in range(8)
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=8, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=8, inner)
        del self.dt_projs
        # 初始化A和D
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=8, merge=True) # (K=8, D, N)
        self.Ds = self.D_init(self.d_inner, copies=8, merge=True) # (K=8, D, N)

        # ss2d
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self._diag_cache = {}

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True #Prevent re-initialization of this bias during model re-initialization
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
    
    def _get_diag_indices(self, H, W,device):
        key = (H, W)
        if key not in self._diag_cache:
            idx_tlbr = get_diagonal_indices(H, W, direction='tlbr')
            idx_trbl = get_diagonal_indices(H, W, direction='trbl')
            self._diag_cache[key] = (idx_tlbr, idx_trbl)
        idx_tlbr,idx_trbl = self._diag_cache[key]
        return idx_tlbr.to(device), idx_trbl.to(device)
    
    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        
        B, C, H, W = x.shape
        L = H * W
        K = 8

        # Original VMamba CrossScan
        # x_hw = x.view(B, -1, L)
        # x_wh = torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)
        # x_hwwh = torch.stack([x_hw, x_wh], dim=1).view(B, 2, -1, L)
        # xs_vmamba = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)  k=4

        #Serpentine
        x_2d = x.view(B, C, H, W)
        x_serp_h = x_2d.clone().contiguous()
        x_serp_h[:, :, 1::2, :] = x_serp_h[:,:,1::2,:].flip(-1) #flip the odd rows
        x_serp_h = x_serp_h.view(B, -1, L)

        x_serp_v = x_2d.clone().contiguous()
        x_serp_v[:, :, :, 1::2] = x_serp_v[:,:,:,1::2].flip(-2) #flip the odd columns
        x_serp_v = torch.transpose(x_serp_v, dim0=2, dim1=3).contiguous().view(B, -1, L)
        xs_serp = torch.stack([x_serp_h, x_serp_v], dim=1).view(B, 2, -1, L)
        xs_serp = torch.cat([xs_serp, torch.flip(xs_serp, dims=[-1])], dim=1) # (b, k, d, l)  k=4

        #Diagonal
        idx_tlbr, idx_trbl = self._get_diag_indices(H, W, x.device)
        x_flat = x.view(B, C, L) # (B,C,L)
        x_diag_tlbr = x_flat[:,:,idx_tlbr] # (B,C,L)
        x_diag_trbl = x_flat[:,:,idx_trbl] # (B,C,L)
        xs_diag = torch.stack([x_diag_tlbr, x_diag_trbl], dim=1).view(B,2,-1,L) # (B,2,C,L)
        xs_diag = torch.cat([xs_diag, torch.flip(xs_diag, dims=[-1])], dim=1) # (b, k, d, l)  k=4

        xs = torch.cat([xs_serp, xs_diag], dim=1) # (b, k, d, l)  k=8

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight) # (b, k, c, l)  c = dt_rank + d_state * 2
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2) # (b, k, dt_rank, l), (b, k, d_state, l), (b, k, d_state, l)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight) # (b, k, d, l)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float


        inv_y_f4 = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        inv_y_b4 = torch.flip(out_y[:, 6:8], dims=[-1]).view(B, 2, -1, L)

        def transpose_back(y):
            return torch.transpose(y.view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        
        def unserp_h(y):
            y_2d = y.clone().view(B, -1, H, W)
            y_2d[:, :, 1::2, :] = y_2d[:, :, 1::2, :].flip(-1) #flip the odd rows back
            return y_2d.view(B, -1, L)
        
        def unserp_v(y):
            y_2d = y.clone().view(B, -1, H, W)
            y_2d[:, :, :, 1::2] = y_2d[:, :, :, 1::2].flip(-2) #flip the odd columns back
            return y_2d.view(B, -1, L)
        
        def undiag(y, idx):
            out = torch.zeros(B, y.shape[1], L, device=y.device, dtype=y.dtype)
            idx_exp = idx.view(1, 1, -1).expand(B, y.shape[1], -1)  # (B, C, L)
            out.scatter_(2, idx_exp, y)  # scatter the values back to their original positions
            return out
        
        #The first four are serpentine, the last four are diagonal
        y0 = out_y[:,0]
        y1 = transpose_back(out_y[:,1])
        y2 = inv_y_f4[:,0]
        y3 = transpose_back(inv_y_f4[:,1])
        y4 = out_y[:,4]
        y5 = out_y[:,5]
        y6 = inv_y_b4[:,0]
        y7 = inv_y_b4[:,1]

        y = y0 + y1 + y2 + y3 + y4 + y5 + y6 + y7

        return y

    def forward(self, x: torch.Tensor):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)  # x is the main input, z is the gating signal

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)  
        y = self.forward_core(x)
        assert y.dtype == torch.float32

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1) #b,c,l -> b,l,c -> b,h,w,c
        y = self.out_norm(y)
        y = y * F.silu(z)   
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out



class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,  # 96
        drop_path: float = 0,  # 0.2
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),  # nn.LN
        attn_drop_rate: float = 0,  # 0
        d_state: int = 16,
        ssm_ratio = 2.0,
        mlp_ratio = 4.0,
        act_layer = nn.GELU,
        mlp_drop = 0.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(hidden_dim)# 96             0.2                   16
        self.ss2d = SS2D(
            d_model=hidden_dim,
            dropout=attn_drop_rate, 
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            )
        
        self.drop_path = DropPath(drop_path)
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)  # 384
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=mlp_drop, channels_first=False)
            
    def _forward(self, input: torch.Tensor):
            x = input + self.drop_path(self.ss2d(self.norm1(input)))
            if self.mlp_branch:
                x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
            return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


class VSSLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(  
        self, 
        dim,  # # 96
        depth,  # 2
        d_state=16,
        drop = 0.,
        attn_drop=0.,
        drop_path=0.,   
        norm_layer=nn.LayerNorm, 
        downsample=None,  # PatchMergin2D
        use_checkpoint=False,  
        mlp_ratio=4.0,
        ssm_ratio=2.0,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        # --- drop_path ---
        if isinstance(drop_path, (list, tuple)):
            dp_list = list(drop_path)
            assert len(dp_list) == depth, \
                f"drop_path length ({len(dp_list)}) must equal depth ({depth})"
        else:
            dp_list = [float(drop_path)] * depth

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,   # 96
                drop_path=dp_list[i],  # 0.2
                norm_layer=norm_layer,  # nn.LN
                attn_drop_rate=attn_drop, # 0 
                d_state=d_state,  # 16
                ssm_ratio=ssm_ratio,
                mlp_ratio=mlp_ratio,
                mlp_drop = 0.0,
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
                x = blk(x)
        Tofusion = x
        if self.downsample is not None:
            x = self.downsample(x)

        return x , Tofusion
    
    
class VSSMEncoder(nn.Module):
    def __init__(self,
                 patch_size=4,
                 in_chans=3,
                 depths=[2, 2, 9, 2],
                 dims=[48, 96, 192, 384],
                 d_state=16,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 ssm_ratio=2.0,
                 mlp_ratio=4.0,
                 ):
        
        super().__init__()

        self.num_layers = len(depths)
        self.dims = dims
        self.embed_dim = dims[0]

        # patch embedding: Input (B, C, H, W) -> (B, H/patch, W/patch, C_embed)
        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        # drop_path probabilities for each block
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # encoder stages
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                ssm_ratio=ssm_ratio,
                mlp_ratio=mlp_ratio,
            )
            self.layers.append(layer)
    
    def forward(self, x):
        """
        輸入: x (B,C,H,W)
        輸出: features (list)，包含每個 stage 的輸出
        """
        x = self.patch_embed(x)   # (B,H/patch,W/patch,C)
        x = self.pos_drop(x)

        fusion_feats = []
        for i, layer in enumerate(self.layers):
            x , Tofusion = layer(x)
            fusion_feats.append(Tofusion)   # (B,H/2^i, W/2^i, C_i)

        return fusion_feats

class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1) 
        self.convforavg = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, bias=True),
        )
        self.convformax = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.convforavg(self.avg_pool(x)) + self.convformax(self.max_pool(x))
        return x * self.sigmoid(attn)     # ⊗
    
    

class CAVSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,  # 96
        drop_path: float = 0,  # 0.2
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),  # nn.LN
        attn_drop_rate: float = 0,  # 0
        d_state: int = 16,
        ssm_ratio = 2.0,
        mlp_ratio = 4.0,
        act_layer = nn.GELU,
        mlp_drop = 0.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(hidden_dim)# 96             0.2                   16
        self.ss2d = SS2D(
            d_model=hidden_dim,
            dropout=attn_drop_rate, 
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            )
        self.drop_path = DropPath(drop_path)
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)  # 384
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=mlp_drop, channels_first=False)

        # CA branch：LN → Conv(1x1) → CA
        self.norm3 = norm_layer(hidden_dim)
        self.conv1x1 = nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False)
        self.ca = ChannelAttention(hidden_dim)

    def _forward(self, input: torch.Tensor):
            x = input + self.drop_path(self.ss2d(self.norm1(input)))
            if self.mlp_branch:
                x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN

            # CA branch（LN → Conv → CA → residual）
            z = self.norm3(x).permute(0,3,1,2).contiguous()   # BHWC→BCHW
            z = self.conv1x1(z)
            z = self.ca(z)
            z = z.permute(0,2,3,1).contiguous()               # BCHW→BHWC
            x = x + z
            return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)
    
class DWConv3x3(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.dwconv = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
        self.pwconv = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, bias=False)
        self.ln = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self,x):
        x = x.permute(0,3,1,2) # BHWC -> BCHW
        x = self.dwconv(x)
        x = self.pwconv(x)
        x = self.ln(x)       
        x = x.permute(0,2,3,1) # BCHW -> BHWC
        x = self.act(x)
        return x
    
class VSSLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        upsample=None, 
        use_checkpoint=False, 
        d_state=16,
        block_cls=VSSBlock,
        extra_in_dim=0, 
        ssm_ratio=2.0,
        mlp_ratio=4.0, 
        use_fuse_conv3x3: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        self.need_fuse = extra_in_dim > 0
        self.use_fuse_conv3x3 = use_fuse_conv3x3 and self.need_fuse
        self.fuse_norm = norm_layer(dim) if (self.need_fuse and use_fuse_conv3x3) else None

        self.blocks = nn.ModuleList([
            block_cls(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                ssm_ratio=ssm_ratio,
                mlp_ratio=mlp_ratio,
            )
            for i in range(depth)])
        
        if True: # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

        if self.need_fuse and use_fuse_conv3x3:
            self.fuse1 = DWConv3x3(in_ch=dim + extra_in_dim, out_ch=dim)
            self.fuse_proj = None
        else:
            self.fuse_proj = nn.Linear(dim + extra_in_dim, dim, bias=False) if self.need_fuse else None
            self.fuse1 = None

    
    #RCF
    def forward(self,x,skip=None):
        if self.upsample is not None:
            x = self.upsample(x)

        if self.need_fuse:
            assert skip is not None
            residual = x
            x = torch.cat([x,skip],dim=-1)
            if self.fuse1 is not None:
                x = self.fuse1(x)
                x = x + residual
            else:
                x = self.fuse_proj(x)
        
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x

            

class VSSMDecoder(nn.Module):
    def __init__(self,num_classes=1000,depths_decoder=[2, 9, 2, 2],dims_decoder=[384,192,96,48],enc_dims=[48, 96, 192, 384],
                 d_state=8, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,ssm_ratio=2.0,mlp_ratio=0.0,
                 norm_layer=nn.LayerNorm, patch_norm=True,use_checkpoint=False):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths_decoder)
        self.dims = dims_decoder

        extra_in = [0, enc_dims[2], enc_dims[1], enc_dims[0]]
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            is_last = (i_layer == self.num_layers - 1)

            use_conv3x3 = True if (extra_in[i_layer] > 0) else False  # Using RCF fusion only when there is a skip connection (i.e., not the first stage)
            # use_conv3x3 = False # No RCF fusion, just use linear projection for skip connection fusion
            layer = VSSLayer_up(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                d_state=d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint,
                block_cls=CAVSSBlock if is_last else VSSBlock,   # ← Onlt last stage uses CAVSSBlock, others use VSSBlock
                extra_in_dim=extra_in[i_layer],  # ← Despite the first stage not having a skip connection, we still set extra_in_dim=0 for consistency
                ssm_ratio=ssm_ratio,
                mlp_ratio=mlp_ratio,
                use_fuse_conv3x3=use_conv3x3,
            )
            self.layers_up.append(layer)

        self.final_upsample = Final_PatchExpand2D(dim=dims_decoder[-1], dim_scale=4, norm_layer=norm_layer)
        self.final_conv = nn.Conv2d(dims_decoder[-1]//4, num_classes, kernel_size=1, bias=False)

    def forward(self, fused_feats):
        """
        輸入: fused_feats (list)，包含每個 stage 的輸出
        輸出: x (B,num_classes,H,W)
        """
        x = fused_feats[-1]  # Last stage (B, H/32, W/32, C)

        for i, layer in enumerate(self.layers_up):
            if i == 0:
                x = layer(x, skip=None)
            else:
                skip = fused_feats[-1 - i]   # Corresponding F3, F2, F1
                # print(f"Decoder Stage {i+1}, skip.shape: {skip.shape}", flush=True)
                x = layer(x, skip=skip)
            # print(f"Decoder Stage {i+1}, x.shape: {x.shape}", flush=True)

        # 3) Final upsample and conv to get the final output
        x = self.final_upsample(x)           # (B, H, W, C/4)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.final_conv(x)               # (B, num_classes, H, W)
        return x


#Testing the SS2D module
# import torch
# B, C, H, W = 2, 192, 16, 16
# x = torch.randn(B, C, H, W, device='cuda')
# model = SS2D(d_model=96).cuda()
# out = model(torch.randn(B, H, W, 96, device='cuda'))
# print(out.shape)  # (2, 16, 16, 96)