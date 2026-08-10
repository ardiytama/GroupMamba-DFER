import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class VSSSBlock(nn.Module):
    """
    Visual Single Selective Scan (VSSS) block.
    This serves as the core scanning mechanism for each directional group.
    """
    def __init__(self, dim):
        super().__init__()
        # In a real implementation, this would call the CUDA selective scan kernel (VMamba).
        # We use a mocked linear projection for structural placeholder.
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x):
        # x shape: (B, L, C)
        return self.norm(self.act(self.proj(x)))

class CAMGate(nn.Module):
    """
    Channel Affinity Modulation (CAM) Gate.
    Dynamically re-weights and fuses the 6 orthogonal channel groups.
    """
    def __init__(self, total_dim, groups=6):
        super().__init__()
        self.groups = groups
        self.group_dim = total_dim // groups
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(total_dim, total_dim // 4)
        self.fc2 = nn.Linear(total_dim // 4, groups * self.group_dim)
        self.act = nn.GELU()

    def forward(self, x_groups):
        # x_groups is a list of 6 tensors, each (B, L, C/6)
        x_concat = torch.cat(x_groups, dim=-1)  # (B, L, C)
        
        # Global pooling over sequence length
        b, l, c = x_concat.shape
        x_pool = x_concat.mean(dim=1)  # (B, C)
        
        # Excitation
        weights = torch.sigmoid(self.fc2(self.act(self.fc1(x_pool))))  # (B, C)
        weights = weights.unsqueeze(1)  # (B, 1, C)
        
        # Modulate
        out = x_concat * weights
        return out

class ModulatedGroupMambaLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.groups = 6
        assert dim % self.groups == 0, "Dimension must be divisible by 6 for GroupMamba"
        self.group_dim = dim // self.groups
        
        # 4 Spatial + 2 Temporal VSSS paths
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
        
        # Split into 6 groups (Channel Partitioning)
        x_split = torch.split(x, self.group_dim, dim=-1)
        
        # Independent scanning (simplified, normally requires specific sequence routing)
        x_out = [self.vsss_paths[i](x_split[i]) for i in range(self.groups)]
        
        # CAM Fusion
        x = self.cam_gate(x_out)
        x = x + res
        
        # FFN
        x = x + self.mlp(self.norm2(x))
        return x

class GReFELModule(nn.Module):
    """
    Geometry-Aware Reliable Facial Expression Learning (GReFEL) Module.
    Blends predicted probability with geometric anchor voting based on Shannon Entropy.
    """
    def __init__(self, dim, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.classifier = nn.Linear(dim, num_classes)
        
        # Trainable geometric anchors for each emotion class
        self.anchors = nn.Parameter(torch.randn(num_classes, dim))
        nn.init.xavier_uniform_(self.anchors)

    def forward(self, features):
        # features: (B, D)
        
        # 1. Primary Classification
        logits = self.classifier(features)
        probs = F.softmax(logits, dim=-1)
        
        # 2. Normalized Shannon Entropy (Uncertainty Estimation)
        # H(P) = -sum(P * log(P)) / log(N)
        epsilon = 1e-7
        entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=-1)
        norm_entropy = entropy / math.log(self.num_classes)
        uncertainty = norm_entropy.unsqueeze(-1)  # (B, 1)
        
        # 3. Geometric Vote (Cosine Similarity against Anchors)
        features_norm = F.normalize(features, p=2, dim=-1)
        anchors_norm = F.normalize(self.anchors, p=2, dim=-1)
        # (B, D) @ (D, C) -> (B, C)
        geom_sim = torch.matmul(features_norm, anchors_norm.T)
        geom_probs = F.softmax(geom_sim, dim=-1)
        
        # 4. Reliability Blending
        # Reliable = (1 - U) * Probs + U * Geom_Probs
        blended_probs = (1.0 - uncertainty) * probs + uncertainty * geom_probs
        
        # Return log_probs for NLLLoss, and features for Triple Loss
        log_probs = torch.log(blended_probs + epsilon)
        return log_probs, features

class GMGReFELVideo(nn.Module):
    """
    GM-GReFEL: A Geometry-Aware Spatiotemporal State-Space Architecture
    """
    def __init__(self, num_classes=7, dim=384, depth=12, num_frames=16, patch_size=16):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        
        # 3D Tubelet Tokenization (S2D logic)
        self.patch_embed = nn.Conv3d(
            in_channels=3, out_channels=dim,
            kernel_size=(2, patch_size, patch_size),
            stride=(2, patch_size, patch_size)
        )
        
        # Modulated GroupMamba Backbone
        self.blocks = nn.ModuleList([
            ModulatedGroupMambaLayer(dim) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)
        
        # GReFEL Head
        self.grefel = GReFELModule(dim, num_classes)

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.patch_embed(x)  # (B, D, T', H', W')
        B, D, T, H, W = x.shape
        
        # Flatten to sequence
        x = x.flatten(2).transpose(1, 2)  # (B, L, D)
        
        # Apply GroupMamba blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        
        # Global Average Pooling for classification
        global_features = x.mean(dim=1)  # (B, D)
        
        # GReFEL Reliability Module
        log_probs, features = self.grefel(global_features)
        
        return log_probs, features

# Quick sanity check logic if run directly
if __name__ == "__main__":
    model = GMGReFELVideo(num_classes=7, dim=384, depth=4)
    print("Model initialized successfully.")
    
    # Dummy input: Batch=2, Channels=3, Frames=16, Resolution=224x224
    dummy_video = torch.randn(2, 3, 16, 224, 224)
    log_probs, features = model(dummy_video)
    
    print("Output Log Probs Shape:", log_probs.shape)
    print("Output Features Shape:", features.shape)
