import math, torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# unet_plus: ConvNeXt-style building blocks
# =========================
class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm equivalent (safe for NCHW)."""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias   = nn.Parameter(torch.zeros(num_channels))
        self.eps    = eps

    def forward(self, x):
        # x: (N,C,H,W)
        u = x.mean(dim=(2,3), keepdim=True)
        s = (x - u).pow(2).mean(dim=(2,3), keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:,None,None] * x + self.bias[:,None,None]

class ConvNeXtBlock(nn.Module):
    """
    Minimal ConvNeXt block:
      - 7x7 depthwise conv
      - LayerNorm
      - 1x1 -> GELU -> 1x1 (channel MLP)
      - optional LayerScale
    """
    def __init__(self, dim, mlp_ratio=4, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm   = LayerNorm2d(dim)
        hidden_dim  = int(dim * mlp_ratio)
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act     = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.gamma   = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma[:,None,None] * x
        return x + shortcut

def convnext_stage(in_ch, out_ch, depth):
    layers = [nn.Conv2d(in_ch, out_ch, 1)]
    for _ in range(depth):
        layers.append(ConvNeXtBlock(out_ch))
    return nn.Sequential(*layers)

class ConvNeXtDown(nn.Module):
    def __init__(self, in_ch, out_ch, depth=2):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.body = convnext_stage(in_ch, out_ch, depth)
    def forward(self, x):
        return self.body(self.pool(x))

class ConvNeXtUp(nn.Module):
    def __init__(self, in_ch, out_ch, depth=2):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
        self.fuse = nn.Conv2d(in_ch, out_ch, 1)
        self.body = convnext_stage(out_ch, out_ch, depth)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad if needed (odd sizes)
        diffY, diffX = x2.size(2)-x1.size(2), x2.size(3)-x1.size(3)
        x1 = F.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        x  = torch.cat([x2, x1], dim=1)
        x  = self.fuse(x)
        return self.body(x)


# ---------- 通用 Stage（与 convnext_stage 对齐） ----------
def physics_stage(in_ch: int,
                  out_ch: int,
                  depth: int,
                  *,
                  kind: str = "heat",
                  downsample: bool = False,
                  steps_per_block: int = 1,
                  dt: float = 0.15,
                  **op_kwargs) -> nn.Sequential:
    """
    统一接口：physics_stage(in_ch, out_ch, depth, kind=..., ...)
      kind {'heat','pm','shock','rd','helmholtz'}
    """
    # 构造算子
    def build_op(ch):
        k = kind.lower()
        if k == "heat":       return HeatOp(ch)
        if k == "pm":         return PMOp(ch, **op_kwargs)
        if k == "shock":      return ShockOp(**op_kwargs)
        if k == "rd":         return RDOp(ch, **op_kwargs)
        if k == "helmholtz":  return HelmholtzSmoothOp(**op_kwargs)
        raise ValueError(f"Unknown physics kind: {kind}")

    layers = [nn.Conv2d(in_ch, out_ch, 1)]
    op = build_op(out_ch)
    for _ in range(depth):
        layers.append(PhysicsBlock(out_ch, op, steps=steps_per_block, dt=dt))
    if downsample:
        layers.append(nn.AvgPool2d(2))
    return nn.Sequential(*layers)


# ----------------------------- factory -----------------------------
DOWN_REGISTRY = {
    "convnext": ConvNeXtDown, # kwargs: depth
}
UP_REGISTRY = {
    "convnext": ConvNeXtUp, # kwargs: depth
}
BLOCK_REGISTRY = {
    "convnext":convnext_stage,
}


def make_down(name: str, in_ch: int, out_ch: int, **kwargs) -> nn.Module:
    name = name.lower()
    if name not in DOWN_REGISTRY:
        raise ValueError(f"Unknown down type '{name}'. Available: {list(DOWN_REGISTRY.keys())}")
    return DOWN_REGISTRY[name](in_ch, out_ch, **kwargs)

def make_up(name: str, in_ch: int, out_ch: int, **kwargs) -> nn.Module:
    name = name.lower()
    if name not in UP_REGISTRY:
        raise ValueError(f"Unknown up type '{name}'. Available: {list(UP_REGISTRY.keys())}")
    if name == "poisson":
        skip_ch = kwargs.pop("skip_ch")
        return UP_REGISTRY[name](in_ch, skip_ch, out_ch, **kwargs)
    return UP_REGISTRY[name](in_ch, out_ch, **kwargs)

def make_block(name: str, in_ch: int, out_ch: int, **kwargs):
    name = name.lower()
    if name not in BLOCK_REGISTRY:
        raise ValueError(f"Unknown down type '{name}'. Available: {list(BLOCK_REGISTRY.keys())}")
    return BLOCK_REGISTRY[name](in_ch, out_ch, **kwargs)
# ----------------------------- sanity test -----------------------------
if __name__ == "__main__":
    # 与 convnext_stage 完全可互换
    stage = physics_stage(64, 128, depth=3, kind="heat", dt=0.15)  # 扩散
    stage = physics_stage(64, 128, depth=3, kind="pm", sigma=0.12, dt=0.2)  # PM 各向异性
    stage = physics_stage(64, 128, depth=3, kind="shock", dt=0.1)  # shock 锐化
    stage = physics_stage(64, 128, depth=3, kind="rd", a_init=0.0, b_init=0.1)  # 反应扩散
    stage = physics_stage(64, 128, depth=3, kind="helmholtz", alpha=0.6)  # 稳定平滑

    x = torch.randn(2, 32, 128, 128)

    # DOWN
    d12 = make_down("convnext", 32, 64)


    # UP (single-skip)
    u_convnext = make_up("convnext", 64, 32)
    up = u_convnext(d12(x), torch.randn(2, 32, 128, 128)); print(up.shape)



    # UP (two-skip)
    # PoissonCrossUp expects two skips; you can wrap it in your Up class to pass (skip, deep_skip)
    #u_xp  = make_up("xpoisson", 64, 32, skip1_ch=32, skip2_ch=64, iters=6, tau=0.2, w=0.5)
    #up = u_xp(d7(x), torch.randn(2, 32, 128, 128), torch.randn(2, 64, 128, 128));print(up.shape)
