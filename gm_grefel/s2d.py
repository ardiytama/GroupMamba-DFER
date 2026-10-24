import torch

SPATIAL_GROUPS = (0, 1, 2, 3)
TEMPORAL_GROUPS = (4, 5)


def load_s2d_checkpoint(model, state_dict, tubelet_t=2):
    # W3D(t,c,h,w) = W2D(c,h,w) / Pt; spatial groups and anchors are inherited,
    # temporal groups keep their init A ~ N(0, 1e-4)
    own = model.state_dict()
    state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    w = state_dict.get('patch_embed.weight')
    if w is not None and w.ndim == 4:
        state_dict['patch_embed.weight'] = w.unsqueeze(2).repeat(1, 1, tubelet_t, 1, 1) / tubelet_t
    inherited = {
        k: v for k, v in state_dict.items()
        if k in own and own[k].shape == v.shape
        and not any(f'vsss_paths.{g}.' in k for g in TEMPORAL_GROUPS)
    }
    required = [k for k in own if k == 'grefel.anchors' or any(f'vsss_paths.{g}.' in k for g in SPATIAL_GROUPS)]
    missing = [k for k in required if k not in inherited]
    assert not missing, f"S2D: not inherited from 2D checkpoint (key or shape mismatch): {missing[:5]}"
    model.load_state_dict(inherited, strict=False)
    print(f"S2D: inherited {len(inherited)} tensors from 2D model")
    return set(inherited)


def build_param_groups(model, inherited, weight_decay):
    # lr_mult is set per stage by set_stage(); inherited params and anchors are damped in warm-up
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        decay = p.ndim >= 2 and not any(k in n for k in ('pos_embed', 'A_log', '.A', '.D', 'anchors'))
        key = (n in inherited, decay)
        groups.setdefault(key, []).append(p)
    return [
        {'params': ps, 'weight_decay': weight_decay if decay else 0.0, 'inherited': inh, 'lr_mult': 1.0}
        for (inh, decay), ps in groups.items()
    ]


def set_stage(optimizer, stage, gamma_freeze=0.01):
    # stage 1: warm-up (inherited damped by gamma_freeze), stage 2: end-to-end fine-tuning
    for g in optimizer.param_groups:
        g['lr_mult'] = gamma_freeze if (stage == 1 and g['inherited']) else 1.0
