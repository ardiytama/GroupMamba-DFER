import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from gm_grefel.model import *
from gm_grefel.model import gm_grefel_base
from gm_grefel.s2d import load_s2d_checkpoint
from timm.models.registry import register_model

register_model(gm_grefel_base)
