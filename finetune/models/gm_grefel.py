import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.registry import register_model

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    print("Warning: mamba_ssm not found. Using a fallback linear/conv layer for VSSSBlock. Please install mamba_ssm for full performance.")

class VSSSBlock(nn.Module):
    """
    Visual Single Selective Scan (VSSS) block.
    """
    def __init__(self, dim, d_state=16):
        super().__init__()
        self.dim = dim
        if HAS_MAMBA:
            self.mamba = Mamba(
                d_model=dim, 
                d_state=d_state,  
                d_conv=4,    
                expand=2,    
            )
        else:
            self.mamba = nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.SiLU(),
                nn.Linear(dim * 2, dim)
            )

    def forward(self, x):
        # x shape: (B, L, C)
        return self.mamba(x)

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
        self.act = nn.GELU()

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
    def __init__(self, dim, num_frames=16, patch_size=16, input_size=224):
        super().__init__()
        self.groups = 6
        assert dim % self.groups == 0, "Dimension must be divisible by 6 for GroupMamba"
        self.group_dim = dim // self.groups
        
        # Calculate grid size (T, H, W)
        self.T = num_frames // 2  # Tubelet size = 2 usually for patch_embed
        self.H = input_size // patch_size
        self.W = input_size // patch_size
        
        self.vsss_paths = nn.ModuleList([
            VSSSBlock(self.group_dim) for _ in range(self.groups)
        ])
        
        self.cam_gate = CAMGate(dim, groups=self.groups)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        res = x
        x = self.norm1(x)
        
        B, L, C = x.shape
        # x is (B, T*H*W, C). Reshape to 3D grid
        T, H, W = self.T, self.H, self.W
        
        # Verify shape
        if L == T * H * W:
            x_grid = x.view(B, T, H, W, C)
        else:
            # Fallback if dimensions don't match perfectly
            x_grid = x.view(B, -1, 1, 1, C)
            T, H, W = x_grid.shape[1:4]
            
        x_split = torch.split(x_grid, self.group_dim, dim=-1)
        
        # Define the 6 scanning directions
        # 1. Spatial Forward (T, H, W)
        # 2. Spatial Backward (T, H, W) reversed
        # 3. Spatial Vertical (T, W, H)
        # 4. Spatial Vertical Backward (T, W, H) reversed
        # 5. Temporal Forward (H, W, T)
        # 6. Temporal Backward (H, W, T) reversed
        
        scan_inputs = []
        
        # Group 0: Spatial Forward
        scan_inputs.append(x_split[0].reshape(B, -1, self.group_dim))
        
        # Group 1: Spatial Backward
        scan_inputs.append(x_split[1].reshape(B, -1, self.group_dim).flip(dims=[1]))
        
        # Group 2: Spatial Vertical Forward
        x2 = x_split[2].transpose(2, 3).contiguous() # (B, T, W, H, C')
        scan_inputs.append(x2.reshape(B, -1, self.group_dim))
        
        # Group 3: Spatial Vertical Backward
        x3 = x_split[3].transpose(2, 3).contiguous()
        scan_inputs.append(x3.reshape(B, -1, self.group_dim).flip(dims=[1]))
        
        # Group 4: Temporal Forward
        x4 = x_split[4].permute(0, 2, 3, 1, 4).contiguous() # (B, H, W, T, C')
        scan_inputs.append(x4.reshape(B, -1, self.group_dim))
        
        # Group 5: Temporal Backward
        x5 = x_split[5].permute(0, 2, 3, 1, 4).contiguous()
        scan_inputs.append(x5.reshape(B, -1, self.group_dim).flip(dims=[1]))
        
        # Apply VSSS Blocks
        x_out = [self.vsss_paths[i](scan_inputs[i]) for i in range(self.groups)]
        
        # Unpermute outputs to original (B, T*H*W, C')
        unpermuted = []
        
        # Group 0
        unpermuted.append(x_out[0])
        
        # Group 1
        unpermuted.append(x_out[1].flip(dims=[1]))
        
        # Group 2
        o2 = x_out[2].view(B, T, W, H, self.group_dim).transpose(2, 3).contiguous()
        unpermuted.append(o2.reshape(B, -1, self.group_dim))
        
        # Group 3
        o3 = x_out[3].flip(dims=[1]).view(B, T, W, H, self.group_dim).transpose(2, 3).contiguous()
        unpermuted.append(o3.reshape(B, -1, self.group_dim))
        
        # Group 4
        o4 = x_out[4].view(B, H, W, T, self.group_dim).permute(0, 3, 1, 2, 4).contiguous()
        unpermuted.append(o4.reshape(B, -1, self.group_dim))
        
        # Group 5
        o5 = x_out[5].flip(dims=[1]).view(B, H, W, T, self.group_dim).permute(0, 3, 1, 2, 4).contiguous()
        unpermuted.append(o5.reshape(B, -1, self.group_dim))
        
        x = self.cam_gate(unpermuted)
        x = x + res
        
        x = x + self.mlp(self.norm2(x))
        return x

class GReFELModule(nn.Module):
    """
    Geometry-Aware Reliable Facial Expression Learning (GReFEL) Module.
    """
    def __init__(self, dim, num_classes):
        super().__init__()
        self.num_classes = num_classes
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
        
        geom_sim = torch.matmul(features_norm, anchors_norm.T) / 0.1 # temperature scaling
        geom_probs = F.softmax(geom_sim, dim=-1)
        
        blended_probs = (1.0 - uncertainty) * probs + uncertainty * geom_probs
        
        log_probs = torch.log(blended_probs + epsilon)
        
        # Return log_probs, raw features, and anchors to be used in Triple Loss
        return log_probs, features, self.anchors

class GMGReFELVideo(nn.Module):
    """
    GM-GReFEL: A Geometry-Aware Spatiotemporal State-Space Architecture
    """
    def __init__(self, num_classes=7, dim=384, depth=12, num_frames=16, patch_size=16):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        
        self.patch_embed = nn.Conv3d(
            in_channels=3, out_channels=dim,
            kernel_size=(2, patch_size, patch_size),
            stride=(2, patch_size, patch_size)
        )
        
        self.blocks = nn.ModuleList([
            ModulatedGroupMambaLayer(dim, num_frames=num_frames, patch_size=patch_size, input_size=224) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.grefel = GReFELModule(dim, num_classes)

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
        x = x.flatten(2).transpose(1, 2)  
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        global_features = x.mean(dim=1)  
        
        log_probs, features, anchors = self.grefel(global_features)
        
        if self.training:
            return log_probs, features, anchors
        return log_probs

@register_model
def gm_grefel_base(pretrained=False, num_classes=7, **kwargs):
    model = GMGReFELVideo(num_classes=num_classes, dim=768, depth=12, num_frames=16, patch_size=16)
    return model

@register_model
def gm_grefel_small(pretrained=False, num_classes=7, **kwargs):
    model = GMGReFELVideo(num_classes=num_classes, dim=384, depth=12, num_frames=16, patch_size=16)
    return model

@register_model
def gm_grefel_tiny(pretrained=False, num_classes=7, **kwargs):
    model = GMGReFELVideo(num_classes=num_classes, dim=192, depth=12, num_frames=16, patch_size=16)
    return model
