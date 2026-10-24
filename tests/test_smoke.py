import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gm_grefel import GMGReFELVideo, TripleLoss, load_pretrained, save_checkpoint
from gm_grefel.data import sample_indices
from gm_grefel.model import selective_scan_zoh
from gm_grefel.s2d import build_param_groups, load_s2d_checkpoint, set_stage


def small():
    return GMGReFELVideo(num_classes=7, dim=96, depth=1, num_frames=4, input_size=32)


def test_zoh_scan_matches_recurrence():
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


def test_paper_shapes():
    m = GMGReFELVideo()
    assert m.pos_embed.shape == (1, 25088, 96) and m.feat_dim == 768 and len(m.blocks) == 3


def test_forward_and_triple_loss():
    m = small().train()
    out = m(torch.randn(2, 3, 4, 32, 32))
    assert out[0].shape == (2, 7) and out[1].shape == (2, 2 * 96) and out[2].shape == (7, 2 * 96)
    loss, parts = TripleLoss()(out, torch.tensor([1, 3]))
    loss.backward()
    assert m.grefel.anchors.grad is not None and set(parts) == {'cls', 'anchor', 'center'}


def test_s2d_transfer_and_stages():
    src = small()
    sd = {k: v.clone() for k, v in src.state_dict().items()}
    sd['patch_embed.weight'] = sd['patch_embed.weight'][:, :, 0]
    m = small()
    temporal = m.blocks[0].vsss_paths[4].A.clone()
    inherited = load_s2d_checkpoint(m, sd)
    assert torch.allclose(m.patch_embed.weight[:, :, 1], sd['patch_embed.weight'] / 2)
    assert torch.equal(m.blocks[0].vsss_paths[0].A_log, src.blocks[0].vsss_paths[0].A_log)
    assert torch.equal(m.blocks[0].vsss_paths[4].A, temporal) and 'grefel.anchors' in inherited
    opt = torch.optim.AdamW(build_param_groups(m, inherited, 0.05), lr=1e-4)
    set_stage(opt, 1, 0.01)
    assert {g['lr_mult'] for g in opt.param_groups if g['inherited']} == {0.01}
    assert {g['lr_mult'] for g in opt.param_groups if not g['inherited']} == {1.0}
    set_stage(opt, 2, 0.01)
    assert {g['lr_mult'] for g in opt.param_groups} == {1.0}


def test_frame_sampling():
    assert sample_indices(16, 16, train=False) == list(range(16))
    assert sample_indices(16, 16, train=True) == list(range(16))
    idx = sample_indices(100, 16, train=False)
    assert len(idx) == 16 and idx == sorted(idx) and max(idx) < 100
    assert sample_indices(5, 16) and max(sample_indices(5, 16)) == 4


def test_checkpoint_roundtrip(tmp_path=None):
    path = os.path.join(tmp_path or '/tmp', 'gm_grefel_test.pth')
    a, b = small(), small()
    save_checkpoint(path, a, dataset='DFEW', uar=0.5)
    meta = load_pretrained(b, path)
    assert meta['dataset'] == 'DFEW' and all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


def test_temporal_scan_stable_with_positive_A():
    x, delta = torch.randn(1, 64, 4), F.softplus(torch.randn(1, 64, 4) * 5)
    A = torch.ones(4, 3, requires_grad=True)
    y = selective_scan_zoh(x, delta, A, torch.randn(1, 64, 3), torch.randn(1, 64, 3), torch.ones(4))
    y.sum().backward()
    assert torch.isfinite(y).all() and torch.isfinite(A.grad).all()


def test_no_weight_decay_on_ssm_and_anchors():
    m = small()
    groups = build_param_groups(m, set(), 0.05)
    nodecay = {id(p) for g in groups if g['weight_decay'] == 0 for p in g['params']}
    named = dict(m.named_parameters())
    for n, p in named.items():
        if any(k in n for k in ('A_log', '.A', '.D', 'anchors', 'pos_embed')) or p.ndim < 2:
            assert id(p) in nodecay, n
    assert id(named['patch_embed.weight']) not in nodecay


def test_s2d_fails_loudly_on_mismatch():
    sd = {k: v for k, v in small().state_dict().items() if k != 'grefel.anchors'}
    try:
        load_s2d_checkpoint(small(), sd)
    except AssertionError:
        return
    raise AssertionError('expected S2D to reject a checkpoint without anchors')


def test_resume_state_roundtrip(tmp_path=None):
    path = os.path.join(tmp_path or '/tmp', 'gm_grefel_resume.pth')
    m = small()
    opt = torch.optim.AdamW(build_param_groups(m, set(), 0.05), lr=1e-4)
    m(torch.randn(1, 3, 4, 32, 32))[0].sum().backward()
    opt.step()
    save_checkpoint(path, m, state=dict(opt=opt.state_dict(), epoch=3, step=12, best_uar=0.4, inherited=[]), uar=0.4)
    ck = torch.load(path)
    opt2 = torch.optim.AdamW(build_param_groups(small(), set(), 0.05), lr=1e-4)
    opt2.load_state_dict(ck['opt'])
    assert ck['epoch'] == 3 and ck['step'] == 12 and len(opt2.state) == len(opt.state)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
