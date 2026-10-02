import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.registry import register_model


def selective_scan_zoh(x, delta, A, B, C, D):
    # x, delta: (B, L, C), A: (C, N), B, C: (B, L, N)
    # A_bar = exp(dA), B_bar = (dA)^-1 (exp(dA) - I) * delta * B
    dA = delta.unsqueeze(-1) * A
    A_bar = torch.exp(dA)
    small = A.abs() < 1e-6
    A_safe = torch.where(small, torch.ones_like(A), A)
    zoh = torch.where(small, delta.unsqueeze(-1).expand_as(dA), torch.expm1(dA) / A_safe)
    h = zoh * B.unsqueeze(2) * x.unsqueeze(-1)
    # h_t = A_bar_t * h_(t-1) + B_bar_t * x_t, parallel doubling scan
    L = x.shape[1]
    k = 1
    while k < L:
        h = torch.cat([h[:, :k], h[:, k:] + A_bar[:, k:] * h[:, :-k]], dim=1)
        A_bar = torch.cat([A_bar[:, :k], A_bar[:, k:] * A_bar[:, :-k]], dim=1)
        k *= 2
    y = torch.einsum('bldn,bln->bld', h, C)
    return y + x * D


class VSSSBlock(nn.Module):
    """
    Visual Single Selective Scan (VSSS) block.
    """
    def __init__(self, dim, d_state=16, temporal=False):
        super().__init__()
        self.temporal = temporal
        self.linear_delta = nn.Linear(dim, dim)
        self.linear_B = nn.Linear(dim, d_state, bias=False)
        self.linear_C = nn.Linear(dim, d_state, bias=False)
        if temporal:
            # A_temporal ~ N(0, 1e-4)
            self.A = nn.Parameter(torch.randn(dim, d_state) * 1e-2)
        else:
            self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(dim, 1))
        self.D = nn.Parameter(torch.ones(dim))
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        # x shape: (B, L, C)
        with torch.cuda.amp.autocast(enabled=False):
            xf = x.float()
            A = self.A if self.temporal else -torch.exp(self.A_log)
            delta = F.softplus(self.linear_delta(xf))
            y = selective_scan_zoh(xf, delta, A.float(), self.linear_B(xf), self.linear_C(xf), self.D)
        y = y.to(x.dtype)
        return y + self.ffn(self.norm(y))


def _snake(H, W, transpose, reverse):
    idx = torch.arange(H * W).view(H, W)
    if transpose:
        idx = idx.t().contiguous()
    idx[1::2] = idx[1::2].flip(-1)
    idx = idx.flatten()
    return idx.flip(0) if reverse else idx


class CAMGate(nn.Module):
    """
    Channel Affinity Modulation (CAM) Gate.
    """
    def __init__(self, total_dim, groups=6):
        super().__init__()
        self.groups = groups
        self.group_dim = total_dim // groups

        self.fc1 = nn.Linear(total_dim, total_dim // 4)
        self.fc2 = nn.Linear(total_dim // 4, total_dim)
        self.act = nn.SiLU()

    def forward(self, x_groups):
        # x_groups is a list of 6 tensors, each (B, L, C/6)
        x_concat = torch.cat(x_groups, dim=-1)  # (B, L, C)

        # Global pooling over sequence length
        x_pool = x_concat.mean(dim=1)  # (B, C)

        # Excitation
        weights = torch.sigmoid(self.fc2(self.act(self.fc1(x_pool))))  # (B, C)
        weights = weights.unsqueeze(1)  # (B, 1, C)

        # Modulate
        out = x_concat * weights
        return out


class ModulatedGroupMambaLayer(nn.Module):
    def __init__(self, dim, grid_size=(8, 56, 56), d_state=16, drop=0.5):
        super().__init__()
        self.groups = 6
        assert dim % self.groups == 0, "Dimension must be divisible by 6 for GroupMamba"
        self.group_dim = dim // self.groups
        self.T, self.H, self.W = grid_size

        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim * 2)

        self.vsss_paths = nn.ModuleList([
            VSSSBlock(self.group_dim, d_state=d_state, temporal=(g >= 4)) for g in range(self.groups)
        ])

        # 4 boustrophedon directions
        perms = [_snake(self.H, self.W, transpose=t, reverse=r) for t, r in ((False, False), (False, True), (True, False), (True, True))]
        self.register_buffer('perms', torch.stack(perms), persistent=False)
        self.register_buffer('inv_perms', torch.stack([p.argsort() for p in perms]), persistent=False)

        self.cam_gate = CAMGate(dim, groups=self.groups)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        res = x
        B, L, C = x.shape
        T, H, W, c = self.T, self.H, self.W, self.group_dim
        assert L == T * H * W, f"expected {T}x{H}x{W} tokens, got {L}"

        x, z = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        x_split = torch.split(x.view(B, T, H * W, C), c, dim=-1)

        out = []
        # Group 0-3: Spatial serpentine scans per frame
        for g in range(4):
            xs = x_split[g][:, :, self.perms[g]].reshape(B * T, H * W, c)
            ys = self.vsss_paths[g](xs).view(B, T, H * W, c)[:, :, self.inv_perms[g]]
            out.append(ys.reshape(B, L, c))

        # Group 4-5: Temporal Forward / Backward
        for g, rev in ((4, False), (5, True)):
            xt = x_split[g].permute(0, 2, 1, 3).reshape(B * H * W, T, c)
            if rev:
                xt = xt.flip(1)
            yt = self.vsss_paths[g](xt)
            if rev:
                yt = yt.flip(1)
            out.append(yt.view(B, H * W, T, c).permute(0, 2, 1, 3).reshape(B, L, c))

        # Z_out = (Y_concat * w_c) * SiLU(Z_gate)
        y = self.drop(self.cam_gate(out) * F.silu(z))
        return res + y


class GReFELModule(nn.Module):
    """
    Geometry-Aware Reliable Facial Expression Learning (GReFEL) Module.
    """
    def __init__(self, dim, num_classes, tau=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.tau = tau
        self.classifier = nn.Linear(dim, num_classes)

        self.anchors = nn.Parameter(torch.randn(num_classes, dim))
        nn.init.xavier_uniform_(self.anchors)

    def forward(self, features):
        logits = self.classifier(features)
        probs = F.softmax(logits, dim=-1)

        epsilon = 1e-7
        entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=-1)
        norm_entropy = entropy / math.log(self.num_classes)
        uncertainty = norm_entropy.unsqueeze(-1)

        features_norm = F.normalize(features, p=2, dim=-1)
        anchors_norm = F.normalize(self.anchors, p=2, dim=-1)

        geom_sim = torch.matmul(features_norm, anchors_norm.T) / self.tau # temperature scaling
        geom_probs = F.softmax(geom_sim, dim=-1)

        blended_probs = (1.0 - uncertainty) * probs + uncertainty * geom_probs

        log_probs = torch.log(blended_probs + epsilon)

        # Return log_probs, raw features, and anchors to be used in Triple Loss
        return log_probs, features, self.anchors


class GMGReFELVideo(nn.Module):
    """
    GM-GReFEL: A Geometry-Aware Spatiotemporal State-Space Architecture
    """
    def __init__(self, num_classes=7, dim=96, depth=3, num_frames=16, input_size=224,
                 tubelet=(2, 4, 4), d_state=16, drop=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.grid_size = (num_frames // tubelet[0], input_size // tubelet[1], input_size // tubelet[2])
        num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        self.patch_embed = nn.Conv3d(
            in_channels=3, out_channels=dim,
            kernel_size=tubelet,
            stride=tubelet
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            ModulatedGroupMambaLayer(dim, grid_size=self.grid_size, d_state=d_state, drop=drop) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.feat_dim = self.grid_size[0] * dim
        self.grefel = GReFELModule(self.feat_dim, num_classes)

    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, x):
        # x: (B, C, T, H, W) -> standard input format from videomae transforms
        if x.ndim == 5:
            # Handle videomae transform output shape which is (B, T, C, H, W) typically
            if x.shape[1] == 3 or x.shape[2] == 3:
                # Make sure it is B, C, T, H, W
                if x.shape[2] == 3 and x.shape[1] > 3:
                    x = x.permute(0, 2, 1, 3, 4)

        x = self.patch_embed(x)
        B, D, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        global_features = x.view(B, T, H * W, D).mean(dim=2).flatten(1)  # (B, T*D) = (B, 768)

        log_probs, features, anchors = self.grefel(global_features)

        if self.training:
            return log_probs, features, anchors
        return log_probs


def load_s2d_checkpoint(model, state_dict, tubelet_t=2):
    # W3D(t,c,h,w) = W2D(c,h,w) / Pt, temporal groups keep their init
    own = model.state_dict()
    state_dict = dict(state_dict)
    w = state_dict.get('patch_embed.weight')
    if w is not None and w.ndim == 4:
        state_dict['patch_embed.weight'] = w.unsqueeze(2).repeat(1, 1, tubelet_t, 1, 1) / tubelet_t
    inherited = {
        k: v for k, v in state_dict.items()
        if k in own and own[k].shape == v.shape
        and not any(f'vsss_paths.{g}.' in k for g in (4, 5))
    }
    model.load_state_dict(inherited, strict=False)
    print(f"S2D: inherited {len(inherited)} tensors from 2D model")
    return set(inherited)


@register_model
def gm_grefel_base(pretrained=False, num_classes=7, **kwargs):
    model = GMGReFELVideo(num_classes=num_classes, dim=96, depth=3, num_frames=16, input_size=224)
    return model


if __name__ == '__main__':
    # check scan against sequential recurrence
    torch.manual_seed(0)
    b, l, d, n = 2, 9, 4, 3
    x, delta = torch.randn(b, l, d), F.softplus(torch.randn(b, l, d))
    A, Bm, Cm, Dv = -torch.rand(d, n) - 0.1, torch.randn(b, l, n), torch.randn(b, l, n), torch.randn(d)
    h, ref = torch.zeros(b, d, n), []
    for t in range(l):
        dA = delta[:, t, :, None] * A
        h = torch.exp(dA) * h + (torch.expm1(dA) / A) * Bm[:, t, None, :] * x[:, t, :, None]
        ref.append((h * Cm[:, t, None, :]).sum(-1) + x[:, t] * Dv)
    assert torch.allclose(selective_scan_zoh(x, delta, A, Bm, Cm, Dv), torch.stack(ref, 1), atol=1e-5)

    m = GMGReFELVideo(num_classes=7, dim=96, depth=1, num_frames=4, input_size=32).train()
    out = m(torch.randn(2, 3, 4, 32, 32))
    assert out[0].shape == (2, 7) and out[1].shape == (2, 2 * 96) and out[2].shape == (7, 2 * 96)
    assert torch.allclose(out[0].exp().sum(-1), torch.ones(2), atol=1e-4)
    print("gm_grefel self-check passed")
