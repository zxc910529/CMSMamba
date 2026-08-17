import torch
import torch.nn.functional as F


def saliency_ndvi_loss(
    logits: torch.Tensor,
    ndvi_mask: torch.Tensor,
    seg_mask: torch.Tensor,
    ignore_index: int = 255,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    N^p = M_NDVI^p ⊙ M_fg^p   （⊙ = XOR，both are binary）
    ℒ_Sal = 1/|1-N| * Σ_p (1-N^p) * (M_NDVI^p - M_fg^p)²
 
    N=1 → inconsistent → (1-N)=0 → exclude
    N=0 → consistent   → (1-N)=1 → calculate MSE
 
    M_NDVI^p : NDVI binary mask  {0,1}
    M_fg^p   : argmax(logits) > 0 → 1（foreground）， 0（background)
    N^p      : M_NDVI^p XOR M_fg^p {0,1}
    """
 
    valid = (seg_mask != ignore_index) & (ndvi_mask != ignore_index)                         # [B, H, W], bool
 
    # ---- M_NDVI: binary {0,1} ----
    m_ndvi_binary = ndvi_mask.long()                           # [B, H, W]
 
    # ---- M_fg_binary: argmax > 0(foreground) XOR ----
    m_fg_binary = (torch.argmax(logits, dim=1) > 0).long()    # [B, H, W]
 
    N = (m_ndvi_binary ^ m_fg_binary)                         # [B, H, W]
 
    # ---- (1 - N^p)：consistent=1，inconsistent=0 ----
    weight = (1 - N).float()                                   # [B, H, W]
 
    # For mse
    m_ndvi_cont = ndvi_mask.float()                            # [B, H, W], {0,1}
    probs = F.softmax(logits, dim=1)                           # [B, C, H, W]
    m_fg_cont = 1.0 - probs[:, 0, :, :]                       # [B, H, W], foreground probability
 
    diff_sq = (m_ndvi_cont - m_fg_cont) ** 2                  # [B, H, W]
 
    valid_f = valid.float()
    weighted_diff = weight * diff_sq * valid_f                 # [B, H, W]
 
    # ---- normalization：valid (1-N) sum ----
    norm = (weight * valid_f).sum().clamp_min(eps)
 
    loss = weighted_diff.sum() / norm
    return loss