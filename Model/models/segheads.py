import math, torch
import torch.nn as nn
import torch.nn.functional as F

from .down_up import make_down, make_up, make_block


MODEL_REGISTRY = {}
def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


# -------------------------
# unet_plus: ConvNeXt-style  + gate + skip adapter
# -------------------------
class GatedFusion(nn.Module):
    """
    Produce spatial+channel gate g in [0,1] from concat([f_main, f_aux]).
    x_fused = g * f_main + (1 - g) * f_aux
    """
    def __init__(self, ch, reduction=4):
        super().__init__()
        mid = max(ch // reduction, 8)
        self.conv1 = nn.Conv2d(2*ch, mid, 1, bias=True)
        self.act   = nn.GELU()
        self.conv2 = nn.Conv2d(mid, ch, 1, bias=True)
        self.spatial = nn.Conv2d(2*ch, 1, kernel_size=3, padding=1, bias=True)

    def forward(self, f_main, f_aux):
        # Channel gate
        z = torch.cat([f_main, f_aux], dim=1)                 # (N,2C,H,W)
        ch_gate = torch.sigmoid(self.conv2(self.act(self.conv1(F.adaptive_avg_pool2d(z,1)))))  # (N,C,1,1)
        # Spatial gate
        sp_gate = torch.sigmoid(self.spatial(z))              # (N,1,H,W)
        g = torch.clamp(ch_gate * sp_gate + 1e-6, 0.0, 1.0)   # broadcast to (N,C,H,W)
        x = g * f_main + (1 - g) * f_aux
        return x, g

class AuxSkipAdapter(nn.Module):
    """
    Predict (gamma, beta) from the aux feature at the same scale,
    then modulate main skip: s' = (1+gamma) * s + beta
    Aux-to-Skip Adapter (FiLM-style modulation)
    """
    def __init__(self, ch, reduction=4):
        super().__init__()
        mid = max(ch // reduction, 8)
        self.adaptor = nn.Sequential(
            nn.Conv2d(ch, mid, 1), nn.GELU(),
            nn.Conv2d(mid, 2*ch, 1)  # -> [gamma, beta]
        )

    def forward(self, s_main, f_aux):
        # Resize aux to skip size if needed
        if f_aux.shape[-2:] != s_main.shape[-2:]:
            f_aux = F.interpolate(f_aux, size=s_main.shape[-2:], mode='bilinear', align_corners=False)
        gb = self.adaptor(f_aux)
        C = s_main.size(1)
        gamma, beta = gb[:, :C], gb[:, C:]
        return (1 + gamma) * s_main + beta

@register_model('unet_plus')
class UNetSegHead_plus(nn.Module):
    """
    - ConvNeXt-style stages (structure upgrade)
    - Gated cross-modal fusion before the encoder
    - Aux-to-skip FiLM adapters at all decoder skips
    """
    def __init__(self, base=32, depth_per_stage=(2,2,2,2)):
        super().__init__()
        d1, d2, d3, d4 = depth_per_stage

        # Dual stems
        self.inc_aux  = make_block('convnext',12, base, depth=1)
        self.inc_main = make_block('convnext',3, base, depth=1)

        # Gated fusion
        self.gate = GatedFusion(base)

        # Encoder pyramid (ConvNeXt)
        self.down1 = make_down("convnext", base, base*2, depth=d1)
        self.down2 = make_down("convnext", base*2, base*4, depth=d2)
        self.down3 = make_down("convnext", base*4, base*8, depth=d3)
        self.down4 = make_down("convnext", base*8, base*16, depth=d4)

        # Store aux features at each scale (for adapters)
        self.aux_down1 = make_down("convnext", base, base*2, depth=1)
        self.aux_down2 = make_down("convnext", base*2, base*4, depth=1)
        self.aux_down3 = make_down("convnext", base*4, base*8, depth=1)
        self.aux_down4 = make_down("convnext", base*8, base*16, depth=1)

        self.up1 = make_up("convnext", base*16, base*8, depth=2)
        self.up2 = make_up("convnext", base*8, base*4, depth=2)
        self.up3 = make_up("convnext", base*4, base*2, depth=2)
        self.up4 = make_up("convnext", base*2, base, depth=2)

        # Aux-to-skip adapters
        self.adapt4 = AuxSkipAdapter(base*8)
        self.adapt3 = AuxSkipAdapter(base*4)
        self.adapt2 = AuxSkipAdapter(base*2)
        self.adapt1 = AuxSkipAdapter(base)

        # Head
        self.outc = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x):
        """
        x: (N, 15, H, W)
        0:12 -> aux (mpIF composite)
        12:15 -> main (original RGB-ish or 3 key channels)
        """
        assert x.dim() == 4 and x.size(1) >= 15, "Expected input with >=15 channels (N,C>=15,H,W)"
        x_aux, x_main = x[:, :12], x[:, 12:15]

        # Dual stems
        f_aux0  = self.inc_aux(x_aux)    # (N, base, H, W)
        f_main0 = self.inc_main(x_main)  # (N, base, H, W)

        # Gated fusion before encoder
        x1, gmap = self.gate(f_main0, f_aux0)  # (N, base, H, W), (optional gate map)

        # Encoder (main)
        e2 = self.down1(x1)     # (N, 2B, H/2, W/2)
        e3 = self.down2(e2)     # (N, 4B, H/4, W/4)
        e4 = self.down3(e3)     # (N, 8B, H/8, W/8)
        e5 = self.down4(e4)     # (N,16B, H/16,W/16)

        # Build a parallel aux pyramid for adapters
        a2 = self.aux_down1(f_aux0)
        a3 = self.aux_down2(a2)
        a4 = self.aux_down3(a3)
        a5 = self.aux_down4(a4)

        # Decoder with Aux-to-Skip modulation
        y  = self.up1(e5, self.adapt4(e4, a4))  # skip at 8B
        y  = self.up2(y,  self.adapt3(e3, a3))  # skip at 4B
        y  = self.up3(y,  self.adapt2(e2, a2))  # skip at 2B
        y  = self.up4(y,  self.adapt1(x1, f_aux0))  # skip at B (use fused x1 as main, modulated by aux stem)

        return self.outc(y) # optionally return gate for diagnostics/losses




MODEL_Registry = {'unet_plus': UNetSegHead_plus}