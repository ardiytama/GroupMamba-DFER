import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import DFEWClips, channel_stats
from .losses import TripleLoss
from .model import create_model
from .s2d import build_param_groups, load_s2d_checkpoint, set_stage
from .weights import load_checkpoint, save_checkpoint


def get_args():
    p = argparse.ArgumentParser('GM-GReFEL training / evaluation on DFEW')
    p.add_argument('--model', default='gm_grefel_base')
    p.add_argument('--data_root', required=True, help='DFEW clips, e.g. Clip/clip_224x224_16f')
    p.add_argument('--label_dir', required=True, help='DFEW EmoLabel_DataSplit')
    p.add_argument('--fold', type=int, default=1)
    p.add_argument('--out', default='saved/gm_grefel')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--warmup_epochs', type=int, default=20, help='S2D stage 1 length')
    p.add_argument('--gamma_freeze', type=float, default=0.01)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--accum', type=int, default=8, help='gradient accumulation steps')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--clip_grad', type=float, default=1.0)
    p.add_argument('--min_lr', type=float, default=1e-6)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--lambda_anchor', type=float, default=10.0)
    p.add_argument('--lambda_center', type=float, default=10.0)
    p.add_argument('--s2d_ckpt', default='', help='converged 2D checkpoint (path / manifest key / URL)')
    p.add_argument('--resume', default='', help='continue from a last.pth (restores optimizer, scaler, epoch) or start from weights (path / manifest key / URL)')
    p.add_argument('--eval_only', action='store_true')
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, num_classes=7):
    model.eval()
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for x, y in loader:
        with torch.cuda.amp.autocast():
            pred = model(x.to(device, non_blocking=True)).argmax(-1).cpu().numpy()
        for t, p in zip(y.numpy(), pred):
            conf[t, p] += 1
    recall = np.diag(conf) / np.maximum(conf.sum(1), 1)
    return recall.mean(), np.diag(conf).sum() / conf.sum(), conf


def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    split = lambda kind: os.path.join(args.label_dir, f'{kind}(single-labeled)', f'set_{args.fold}.csv')
    os.makedirs(args.out, exist_ok=True)

    train_set = DFEWClips(args.data_root, split('train'), train=True)
    mean, std = channel_stats(train_set, cache=os.path.join(args.out, f'channel_stats_fold{args.fold}.json'))
    train_set.mean, train_set.std = torch.tensor(mean).view(3, 1, 1, 1), torch.tensor(std).view(3, 1, 1, 1)
    test_set = DFEWClips(args.data_root, split('test'), train=False, mean=mean, std=std)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, args.batch_size, num_workers=args.num_workers, pin_memory=True)

    model = create_model(args.model)
    inherited, state = set(), None
    if args.resume:
        state = load_checkpoint(args.resume)
        model.load_state_dict(state['model'])
        inherited = set(state.get('inherited', []))
    elif args.s2d_ckpt:
        ckpt = load_checkpoint(args.s2d_ckpt)
        inherited = load_s2d_checkpoint(model, ckpt.get('model', ckpt))
    model.to(device)

    if args.eval_only:
        uar, war, _ = evaluate(model, test_loader, device)
        print(f'UAR {uar * 100:.2f}  WAR {war * 100:.2f}')
        return

    criterion = TripleLoss(args.lambda_anchor, args.lambda_center)
    optimizer = torch.optim.AdamW(build_param_groups(model, inherited, args.weight_decay), lr=args.lr, betas=(0.9, 0.999))
    scaler = torch.cuda.amp.GradScaler()
    steps_per_epoch = len(train_loader) // args.accum
    assert steps_per_epoch >= 1, 'accum is larger than the number of batches per epoch'
    total_steps = args.epochs * steps_per_epoch
    log = open(os.path.join(args.out, 'log.jsonl'), 'a', encoding='utf-8')

    start_epoch, step, best_uar = 0, 0, -1.0
    if state is not None and 'opt' in state:
        optimizer.load_state_dict(state['opt'])
        scaler.load_state_dict(state['scaler'])
        start_epoch, step, best_uar = state['epoch'] + 1, state['step'], state['best_uar']

    for epoch in range(start_epoch, args.epochs):
        set_stage(optimizer, 1 if epoch < args.warmup_epochs else 2, args.gamma_freeze)
        model.train()
        t0, run = time.time(), {'loss': 0.0, 'n': 0}
        for i, (x, y) in enumerate(train_loader):
            if i // args.accum >= steps_per_epoch:
                break
            with torch.cuda.amp.autocast():
                loss, parts = criterion(model(x.to(device, non_blocking=True)), y.to(device, non_blocking=True))
            scaler.scale(loss / args.accum).backward()
            run['loss'] += loss.item()
            run['n'] += 1
            if (i + 1) % args.accum == 0:
                lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * step / total_steps))
                for g in optimizer.param_groups:
                    g['lr'] = lr * g['lr_mult']
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1
        uar, war, conf = evaluate(model, test_loader, device)
        rec = {'epoch': epoch, 'loss': run['loss'] / max(run['n'], 1), 'uar': uar, 'war': war, 'time': time.time() - t0}
        print(json.dumps(rec))
        log.write(json.dumps(rec) + '\n')
        log.flush()
        meta = dict(arch=args.model, dataset='DFEW', fold=args.fold, epoch=epoch, uar=float(uar), war=float(war))
        if uar > best_uar:
            best_uar = uar
            save_checkpoint(os.path.join(args.out, 'best.pth'), model, **meta)
            np.save(os.path.join(args.out, 'best_confusion.npy'), conf)
        resume_state = dict(opt=optimizer.state_dict(), scaler=scaler.state_dict(), epoch=epoch, step=step,
                            best_uar=float(best_uar), inherited=sorted(inherited))
        save_checkpoint(os.path.join(args.out, 'last.pth'), model, state=resume_state, **meta)


if __name__ == '__main__':
    main()
