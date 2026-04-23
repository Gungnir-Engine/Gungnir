"""
train_halfka.py — PyTorch trainer for Gungnir's HalfKAv2_hm small-net arch.

Trains a model with the same architecture as src/nnue.cpp's forward pass.
The forward pass here uses floats throughout; quantization to int8/int16
and the .nnue file writer live in Session 46 (tools/save_nnue.py).

Usage:
    python tools/train_halfka.py data/labels_1M.txt \\
        --out data/model.pt --epochs 2 --batch 256 --max 50000

Output is a PyTorch checkpoint (.pt) — NOT yet a .nnue file. Loading it
into Gungnir requires Session 46's serializer.

Dependencies: torch, numpy.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gungnir_nnue import (
    FT_INPUT_DIM, FT_OUTPUT_DIM, PSQT_BUCKETS, LAYER_STACKS,
    L2, FC0_OUT, FC1_IN_PAD, FC1_OUT, FC2_IN_PAD,
    halfka_features, fen_to_board,
)


# ============================================================================
# Model — HalfKAv2_hm forward pass in float32
# ============================================================================

class GungnirHalfKA(nn.Module):
    def __init__(self):
        super().__init__()
        # Feature transformer: sparse sum via EmbeddingBag.
        self.ft = nn.EmbeddingBag(FT_INPUT_DIM, FT_OUTPUT_DIM, mode='sum')
        self.ft_bias = nn.Parameter(torch.zeros(FT_OUTPUT_DIM))
        # PSQT: 8 buckets × FT_INPUT_DIM.
        self.psqt = nn.EmbeddingBag(FT_INPUT_DIM, PSQT_BUCKETS, mode='sum')

        # 8 layer stacks. Each has its own fc_0 / fc_1 / fc_2.
        self.fc0 = nn.ModuleList([nn.Linear(FT_OUTPUT_DIM, FC0_OUT) for _ in range(LAYER_STACKS)])
        self.fc1 = nn.ModuleList([nn.Linear(FC1_IN_PAD, FC1_OUT) for _ in range(LAYER_STACKS)])
        self.fc2 = nn.ModuleList([nn.Linear(FC2_IN_PAD, 1) for _ in range(LAYER_STACKS)])

        # Init: FT weights small but nonzero, FT bias offset so accumulator
        # starts in the linear regime of the [0, 127] clamp (not saturated,
        # not dead). Without this offset, near-zero accumulator makes the
        # pairwise transform `clamp(a,0,127)*clamp(b,0,127)/128` produce
        # zero gradients early in training.
        nn.init.normal_(self.ft.weight, mean=0.0, std=1.0)
        nn.init.constant_(self.ft_bias, 40.0)
        nn.init.zeros_(self.psqt.weight)
        for stack in (self.fc0, self.fc1, self.fc2):
            for lin in stack:
                nn.init.kaiming_uniform_(lin.weight, a=5**0.5)
                nn.init.zeros_(lin.bias)

    def forward(self, w_feats, w_offsets, b_feats, b_offsets, stm, bucket):
        """
        w_feats, b_feats:  LongTensor [total_feats]   — concatenated feature indices.
        w_offsets, b_offsets: LongTensor [B]           — start offsets into *_feats per sample.
        stm:               LongTensor [B]              — 0 for white, 1 for black.
        bucket:            LongTensor [B]              — 0..7.
        Returns:           FloatTensor [B]             — eval in SF-internal-cp / 16.
        """
        B = stm.shape[0]

        acc_w = self.ft(w_feats, w_offsets) + self.ft_bias          # [B, 128]
        acc_b = self.ft(b_feats, b_offsets) + self.ft_bias          # [B, 128]
        psqt_w = self.psqt(w_feats, w_offsets)                       # [B, 8]
        psqt_b = self.psqt(b_feats, b_offsets)                       # [B, 8]

        # STM perspective first, opponent second.
        stm_f = stm.float().unsqueeze(-1)                           # [B, 1]
        acc_stm = acc_w * (1 - stm_f) + acc_b * stm_f
        acc_opp = acc_b * (1 - stm_f) + acc_w * stm_f

        # Pairwise transform: clamp(a[:H], 0, 127) * clamp(a[H:], 0, 127) / 128
        H = FT_OUTPUT_DIM // 2                                       # 64
        def pairwise(a):
            s0 = torch.clamp(a[:, :H], 0.0, 127.0)
            s1 = torch.clamp(a[:, H:], 0.0, 127.0)
            return s0 * s1 / 128.0
        ft_stm = pairwise(acc_stm)
        ft_opp = pairwise(acc_opp)
        ft = torch.cat([ft_stm, ft_opp], dim=-1)                     # [B, 128]

        # Per-bucket fc_0 / fc_1 / fc_2. Group samples by bucket for efficiency.
        fc0_out = torch.zeros(B, FC0_OUT, device=ft.device)
        fc1_in  = torch.zeros(B, FC1_IN_PAD, device=ft.device)
        fc1_out = torch.zeros(B, FC1_OUT, device=ft.device)
        fc2_in  = torch.zeros(B, FC2_IN_PAD, device=ft.device)
        scalar  = torch.zeros(B, 1, device=ft.device)

        for bk in range(LAYER_STACKS):
            mask = (bucket == bk)
            if not mask.any():
                continue
            idx = mask.nonzero(as_tuple=True)[0]
            x = ft[idx]                                              # [Bk, 128]
            o0 = self.fc0[bk](x)                                     # [Bk, 16]
            fc0_out[idx] = o0

            # sqr-clipped-relu and clipped-relu over first L2 outputs.
            sqr = torch.clamp((o0[:, :L2] ** 2) / (1 << 19), 0.0, 127.0)
            rel = torch.clamp(o0[:, :L2] / (1 << 6), 0.0, 127.0)
            slab = torch.cat([
                sqr, rel,
                torch.zeros(sqr.shape[0], FC1_IN_PAD - 2 * L2, device=ft.device),
            ], dim=-1)                                               # [Bk, 32]
            fc1_in[idx] = slab

            o1 = self.fc1[bk](slab)                                  # [Bk, 32]
            fc1_out[idx] = o1
            ac1 = torch.clamp(o1 / (1 << 6), 0.0, 127.0)
            fc2_in[idx] = ac1

            o2 = self.fc2[bk](ac1)                                   # [Bk, 1]
            scalar[idx] = o2

        # Skip connection: fc_0[:, 15] * 9600/8128
        skip = fc0_out[:, 15:16] * (9600.0 / 8128.0)
        positional = scalar + skip                                   # [B, 1]

        # PSQT: (psqt_stm[bucket] - psqt_opp[bucket]) / 2
        psqt_stm = psqt_w * (1 - stm_f) + psqt_b * stm_f
        psqt_opp = psqt_b * (1 - stm_f) + psqt_w * stm_f
        psqt = (psqt_stm.gather(1, bucket.unsqueeze(-1)) -
                psqt_opp.gather(1, bucket.unsqueeze(-1))) / 2.0       # [B, 1]

        return (psqt + positional).squeeze(-1) / 16.0                # [B]


# ============================================================================
# Dataset — lazy-loads labels.txt, extracts features on demand
# ============================================================================

class LabelsDataset(torch.utils.data.Dataset):
    def __init__(self, path, max_rows=-1):
        self.samples = []   # list of (fen, score_cp)
        print(f"Loading labels from {path}...", flush=True)
        with open(path) as f:
            for i, line in enumerate(f):
                if max_rows > 0 and i >= max_rows:
                    break
                line = line.strip()
                if not line or '|' not in line:
                    continue
                fen, s = line.rsplit('|', 1)
                try:
                    score = int(s)
                except ValueError:
                    continue
                self.samples.append((fen, score))
        print(f"  {len(self.samples)} samples", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fen, score = self.samples[idx]
        board, stm = fen_to_board(fen)
        fw = halfka_features(fen, 0)
        fb = halfka_features(fen, 1)
        bucket = (len(board) - 1) // 4
        if bucket < 0: bucket = 0
        if bucket > 7: bucket = 7
        return fw, fb, stm, bucket, score


def collate(batch):
    w_all, b_all = [], []
    w_off, b_off = [], []
    stms, buckets, scores = [], [], []
    for fw, fb, stm, bk, sc in batch:
        w_off.append(len(w_all))
        w_all.extend(fw)
        b_off.append(len(b_all))
        b_all.extend(fb)
        stms.append(stm)
        buckets.append(bk)
        scores.append(sc)
    return (torch.tensor(w_all, dtype=torch.long),
            torch.tensor(w_off, dtype=torch.long),
            torch.tensor(b_all, dtype=torch.long),
            torch.tensor(b_off, dtype=torch.long),
            torch.tensor(stms, dtype=torch.long),
            torch.tensor(buckets, dtype=torch.long),
            torch.tensor(scores, dtype=torch.float32))


# ============================================================================
# Training loop
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('labels', help='Input FEN|score file.')
    ap.add_argument('--out',    default='data/model.pt')
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch',  type=int, default=256)
    ap.add_argument('--lr',     type=float, default=1e-3)
    ap.add_argument('--max',    type=int, default=-1, help='Use at most N labels (default: all)')
    ap.add_argument('--workers',type=int, default=0)
    ap.add_argument('--save-every', type=int, default=5,
                    help='Save checkpoint every N epochs (default: 5)')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    ds = LabelsDataset(args.labels, max_rows=args.max)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, collate_fn=collate,
    )

    net = GungnirHalfKA().to(device)
    opt = optim.Adam(net.parameters(), lr=args.lr)

    # Texel-style loss: sigmoid(eval / 400) should match sigmoid(target / 400).
    # We compute MSE on the sigmoid-space targets.
    def target_transform(score):
        return torch.sigmoid(score / 400.0)

    print(f"Training {args.epochs} epoch(s), batch={args.batch}, lr={args.lr}, samples={len(ds)}", flush=True)
    for epoch in range(args.epochs):
        net.train()
        t0 = time.time()
        total_loss, n_batches = 0.0, 0
        for batch in loader:
            wf, wo, bf, bo, stm, bk, sc = [t.to(device) for t in batch]
            pred = net(wf, wo, bf, bo, stm, bk)
            # Pred is "internal cp / 16"; multiply back by ~16 to get cp, then / 400 for Texel.
            # In practice train against the sigmoid target directly.
            pred_sig = torch.sigmoid(pred * 16.0 / 400.0)
            target   = target_transform(sc)
            loss = ((pred_sig - target) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
            if n_batches % 100 == 0:
                print(f"  epoch {epoch+1} batch {n_batches} loss={total_loss/n_batches:.6f}",
                      flush=True)
        dt = time.time() - t0
        print(f"epoch {epoch+1}/{args.epochs} done in {dt:.1f}s  avg_loss={total_loss/max(1,n_batches):.6f}",
              flush=True)

        # Periodic checkpoint (survives crashes / allows early inspection).
        if args.save_every > 0 and ((epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1):
            os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
            ckpt_path = args.out if epoch == args.epochs - 1 else \
                        args.out.replace('.pt', f'.e{epoch+1}.pt')
            torch.save(net.state_dict(), ckpt_path)
            print(f"  [checkpoint: {ckpt_path}]", flush=True)

    print(f"Saved final PyTorch checkpoint to {args.out}", flush=True)
    print("Next: run tools/save_nnue.py to convert to a Gungnir-loadable .nnue.", flush=True)


if __name__ == '__main__':
    main()
