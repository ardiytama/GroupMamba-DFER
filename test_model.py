import torch
import sys
sys.path.insert(0, 'finetune')
from models.gm_grefel import gm_grefel_base

model = gm_grefel_base()
model.train()
x = torch.randn(2, 3, 16, 224, 224)
log_probs, features, anchors = model(x)
print(f"Log probs: {log_probs.shape}, Features: {features.shape}, Anchors: {anchors.shape}")
print("Forward pass successful!")
