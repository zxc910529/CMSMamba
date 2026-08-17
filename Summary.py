#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark: Params / MACs / GFLOPs / Memory / FPS
"""
import argparse
import time
 
import torch
import torch.nn as nn
from ptflops import get_model_complexity_info
import types


from models.CMSMamba import CMSMamba



# ──────────────────────────────────────────
# SS2D FLOPs (Hand-calculated)
# ──────────────────────────────────────────

def get_ss2d_params(block):
    """ Detect d_state, d_inner from a block (ss2d or self_attention)"""
    ssm = getattr(block, 'ss2d', None) or getattr(block, 'self_attention', None)
    if ssm is None:
        raise ValueError("Cant find ss2d or self_attention")
    d_state  = ssm.A_logs.shape[1]
    d_inner  = ssm.A_logs.shape[0] // 4   
    return d_state, d_inner

def ss2d_flops_layers(layers, H, W, is_encoder=True):
    """Calculate the SS2D FLOPs for a set of layers and return (flops, H_out, W_out)."""
    total = 0
    for i, layer in enumerate(layers):
        if not is_encoder and i > 0:
            H *= 2; W *= 2
        if not layer.blocks:
            continue
        d_state, d_inner = get_ss2d_params(layer.blocks[0])
        D, N, L = 4 * d_inner, d_state, H * W
        total += len(layer.blocks) * (9 * L * D * N + L * D)
        if is_encoder and i < len(layers) - 1:
            H //= 2; W //= 2
    return total, H, W

def ss2d_flops_cmsmamba(model, H, W, patch=4):
    enc_layers = model.encoder1.layers
    f_enc, He, We = ss2d_flops_layers(enc_layers, H // patch, W // patch)
    f_enc *= 2  # encoder1 + encoder2 
    f_dec, _, _ = ss2d_flops_layers(model.decoder.layers_up, He, We, is_encoder=False)
    return f_enc + f_dec


SS2D_REGISTRY = {
    "cmsmamba":    ss2d_flops_cmsmamba,
}


# ──────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────
 
def parse_unit(s: str) -> float:
    """ Parse a string like "1.23 GMac" or "456.7 M" and return the numeric value in base units (e.g., 1.23e9 or 4.567e8)."""
    if s is None:
        raise RuntimeError("ptflops returned None — forward pass likely crashed.")
    val, *rest = s.strip().split()
    val = float(val)
    unit = rest[0].lower() if rest else ""
    scale = {"k": 1e3, "m": 1e6, "g": 1e9, "t": 1e12}
    for prefix, factor in scale.items():
        if unit.startswith(prefix):
            return val * factor
    return val
 
 
def fmt_g(x: float) -> str:
    return f"{x / 1e9:.3f} G"
 
 
def make_inputs(kind, h, w, c1, c2, device, input_style="kwargs"):
    if kind == "single":
        return [torch.randn(1, c1, h, w, device=device)]
    if input_style == "list":
        # The `forward` method takes `x=[t1, t2]`; wrapping them in a list ensures `model(*inputs)` unpacks them correctly.
        return [[
            torch.randn(1, c1, h, w, device=device),
            torch.randn(1, c2, h, w, device=device),
        ]]
    return [
        torch.randn(1, c1, h, w, device=device),
        torch.randn(1, c2, h, w, device=device),
    ]
 
 
# ──────────────────────────────────────────
# ptflops input constructor (dual)
# ──────────────────────────────────────────
 
def dual_constructor(h, w, c1, c2, device,key1 = "x1", key2="x2"):
    def _ctor(_):
        return {
            key1 : torch.randn(1, c1, h, w, device=device),
            key2 : torch.randn(1, c2, h, w, device=device),
        }
    return _ctor
 
# ──────────────────────────────────────────
# Speed/Memory Benchmark
# ──────────────────────────────────────────
 
def measure_speed(model, inputs, device, warmup=20, iteration=100):
    """回傳 (speed_ms, fps, memory_MB)。"""
    model.eval()
 
    # warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(*inputs) if len(inputs) > 1 else model(inputs[0])
 
    mem_mb = round(torch.cuda.memory_allocated() / 1024 ** 2, 2) if device == "cuda" else 0.0
 
    # timing
    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.time()
    with torch.no_grad():
        for _ in range(iteration):
            model(*inputs) if len(inputs) > 1 else model(inputs[0])
    torch.cuda.synchronize() if device == "cuda" else None
    elapsed = time.time() - t0
 
    speed_ms = elapsed / iteration * 1000
    fps = iteration / elapsed
    return speed_ms, fps, mem_mb
 
 
# ──────────────────────────────────────────
# Model builders
# ──────────────────────────────────────────

def build_cmsmamba(*,in_channels, num_classes, **kw):
    c1, c2 = in_channels if isinstance(in_channels, tuple) else (in_channels, 2)
    return CMSMamba(
            input_channels_1=c1,  # RGB
            input_channels_2=c2,
            num_classes=num_classes,
            depths=[1, 1, 2, 1],
            depths_decoder=[1, 2, 1, 1],
            dims=[48, 96, 192, 384],
            dims_decoder=[384,192,96,48],
            d_state=8,
            use_checkpoint=False,
            ssm_ratio=2.0,
            mlp_ratio=4.0,
            mlp_ratio_decoder=0.0,
            dropout=0.05,
        )


MODEL_REGISTRY = {
    "cmsmamba":        {"kind": "dual", "builder": build_cmsmamba, "keys":("x1", "x2")},
}
 
 
# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser("Segmentation Model Benchmark")
    parser.add_argument("--model",   required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--height",  type=int, default=608)
    parser.add_argument("--width",   type=int, default=608)
    parser.add_argument("--classes", type=int, default=6)
    parser.add_argument("--c1",      type=int, default=3,  help="single: in_channels | dual: First channels")
    parser.add_argument("--c2",      type=int, default=2,  help="dual: Second channels")
    parser.add_argument("--ckpt-resnet", default=None)
    parser.add_argument("--cuda",    action="store_true")
    parser.add_argument("--warmup",  type=int, default=3) #20
    parser.add_argument("--iters",   type=int, default=5) #100
    args = parser.parse_args()
 
    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"

    spec   = MODEL_REGISTRY[args.model]
    keys = spec.get("keys", ("x1", "x2"))
    input_style = spec.get("input_style", "kwargs")
    kind   = spec["kind"]
 
    # in_channels: single → int, dual → tuple
    in_channels = args.c1 if kind == "single" else (args.c1, args.c2)
 
    model = spec["builder"](
        in_channels=in_channels,
        num_classes=args.classes,
        img_size=args.height,
        ckpt_resnet=args.ckpt_resnet,
    ).to(device).eval()
    
    # ── MACs / Params ──

    keys = spec.get("keys", ("x1", "x2"))
    ctor = dual_constructor(args.height, args.width, args.c1, args.c2, device, *keys)

    macs_str, params_str = get_model_complexity_info(
        model, (args.c1, args.height, args.width),
        as_strings=True, print_per_layer_stat=False, verbose=False,
        input_constructor=ctor,
     )
 
    macs  = parse_unit(macs_str)
     # ── SS2D (Hand-calculated)──
    ss2d_fn = SS2D_REGISTRY.get(args.model)
    ss2d_flops = ss2d_fn(model, args.height, args.width) if ss2d_fn else 0
    ss2d_macs  = ss2d_flops / 2.0

    total_macs  = macs + ss2d_macs
    total_flops = 2.0 * total_macs
 
    # ── Speed / Memory ──
    inputs = make_inputs(kind, args.height, args.width, args.c1, args.c2, device,input_style)
    speed_ms, fps, mem_mb = measure_speed(model, inputs, device, args.warmup, args.iters)
 
   # ── Output ──
    ch_info = f"{args.c1}ch" if kind == "single" else f"{args.c1}+{args.c2}ch"
    print("=" * 46)
    print(f"  Model   : {args.model}")
    print(f"  Input   : {ch_info}  {args.height}×{args.width}")
    print("-" * 46)
    print(f"  Params  : {params_str}")
    print(f"  MACs    : {fmt_g(total_macs)}")
    print(f"  GFLOPs  : {fmt_g(total_flops)}")
    if ss2d_fn:
        print(f"    ( SS2D : {fmt_g(ss2d_macs)} MACs / {fmt_g(ss2d_flops)} FLOPs)")
    print(f"  Memory  : {mem_mb} MB" if device == "cuda" else "  Memory  : N/A (CPU)")
    print(f"  Speed   : {speed_ms:.2f} ms/iter")
    print(f"  FPS     : {fps:.2f}")
    print("=" * 46)
 
 



if __name__ == "__main__":
    main()

# Usage example:

# python summary.py --model cmsmamba --c1 3 --c2 2 --height 608 --width 608 --classes 6 --cuda
