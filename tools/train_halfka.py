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
# Model — HalfKAv2_hm forward pass in float32, with Quantization-Aware
# Training (Session 48): the forward pass simulates the int arithmetic that
# src/nnue.cpp does. Weights are fake-quantized via STE so they learn to
# survive the int round-trip.
#
# SF's small-net scheme: QA=255 for FT (int16 storage), QB=64 for layer
# weights (int8 storage). Accumulator values are clamped at 127, which
# (after divide by QA=255) corresponds to float ~0.498. The forward pass
# works in "int-activation space" — i.e., weights and biases are stored as
# floats but represent their int-scaled values directly. So FT bias 40
# (float) "is" int16 value 40, representing float 40/255 ≈ 0.157 in SF's
# natural scale.
# ============================================================================

QA = 255.0      # FT quantization scale (int16 storage)
QB = 64.0       # layer quantization scale (int8 storage)


class _RoundSTE(torch.autograd.Function):
    """Round with a straight-through estimator: forward rounds, backward is identity."""
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad):
        return grad


def round_ste(x):
    return _RoundSTE.apply(x)


def fake_quant(w, bits):
    """Apply fake integer quantization with STE: round to nearest int, clip to
    [-2^(bits-1), 2^(bits-1)-1]. Forward pass uses the quantized value;
    backward pass treats round as identity so gradients flow."""
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    return torch.clamp(round_ste(w), qmin, qmax)


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

        # Init for QAT: values stored as floats but representing int16/int8.
        # Magnitudes chosen so fc_0 outputs typical ~1000-2000 — small enough
        # that sqr-clipped-relu (>>19) and clipped-relu (>>6) produce useful
        # values 2-30, not saturating at 127 and not rounding to 0.
        #
        # FT weights std=5 → rounded int16 values ±5-15 typical.
        # FT bias=30 → keeps accumulator in [0, 127] clamp range.
        # Layer weights std=3 → rounded int8 values ±2-9.
        nn.init.normal_(self.ft.weight, mean=0.0, std=5.0)
        nn.init.constant_(self.ft_bias, 30.0)
        nn.init.zeros_(self.psqt.weight)
        for stack in (self.fc0, self.fc1, self.fc2):
            for lin in stack:
                nn.init.normal_(lin.weight, mean=0.0, std=3.0)
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

        # --- QAT: fake-quantize FT weights/bias and PSQT weights to int16 ---
        # (Since EmbeddingBag doesn't let us swap weights on the fly, we
        # compute the accumulator manually using gather + sum on quantized
        # weights. Slower per-batch but gives us the QAT guarantee.)
        ft_weight_q = fake_quant(self.ft.weight, 16)           # [22528, 128] int16-valued floats
        ft_bias_q   = fake_quant(self.ft_bias,   16)           # [128]
        psqt_w_q    = fake_quant(self.psqt.weight, 32)         # [22528, 8] int32

        def sparse_sum(w, feats, offsets):
            """Sum rows of w at indices feats, grouped into samples by offsets."""
            # offsets: [B], feats: [total_feats].
            # Build a segment-id for each feature (which sample it belongs to).
            gathered = w[feats]                                # [total_feats, D]
            # Create sample index per feature.
            n = feats.shape[0]
            sample_ids = torch.zeros(n, dtype=torch.long, device=feats.device)
            if len(offsets) > 1:
                sample_ids[offsets[1:]] = 1
                sample_ids = sample_ids.cumsum(0)
            out = torch.zeros(B, w.shape[1], device=w.device, dtype=w.dtype)
            out.index_add_(0, sample_ids, gathered)
            return out

        acc_w = sparse_sum(ft_weight_q, w_feats, w_offsets) + ft_bias_q    # [B, 128]
        acc_b = sparse_sum(ft_weight_q, b_feats, b_offsets) + ft_bias_q    # [B, 128]
        psqt_w = sparse_sum(psqt_w_q,  w_feats, w_offsets)                  # [B, 8]
        psqt_b = sparse_sum(psqt_w_q,  b_feats, b_offsets)                  # [B, 8]

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

            # Fake-quantize layer weights to int8.
            fc0_w_q = fake_quant(self.fc0[bk].weight, 8)
            fc0_b_q = fake_quant(self.fc0[bk].bias, 32)              # int32 (QA*QB scale)
            fc1_w_q = fake_quant(self.fc1[bk].weight, 8)
            fc1_b_q = fake_quant(self.fc1[bk].bias, 32)
            fc2_w_q = fake_quant(self.fc2[bk].weight, 8)
            fc2_b_q = fake_quant(self.fc2[bk].bias, 32)

            o0 = torch.nn.functional.linear(x, fc0_w_q, fc0_b_q)     # [Bk, 16]
            fc0_out[idx] = o0

            # sqr-clipped-relu and clipped-relu over first L2 outputs.
            sqr = torch.clamp((o0[:, :L2] ** 2) / (1 << 19), 0.0, 127.0)
            rel = torch.clamp(o0[:, :L2] / (1 << 6), 0.0, 127.0)
            slab = torch.cat([
                sqr, rel,
                torch.zeros(sqr.shape[0], FC1_IN_PAD - 2 * L2, device=ft.device),
            ], dim=-1)                                               # [Bk, 32]
            fc1_in[idx] = slab

            o1 = torch.nn.functional.linear(slab, fc1_w_q, fc1_b_q)  # [Bk, 32]
            fc1_out[idx] = o1
            ac1 = torch.clamp(o1 / (1 << 6), 0.0, 127.0)
            fc2_in[idx] = ac1

            o2 = torch.nn.functional.linear(ac1, fc2_w_q, fc2_b_q)   # [Bk, 1]
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
    ap.add_argument('--warm-start', default=None,
                    help='Load existing checkpoint before training (for fine-tuning).')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    ds = LabelsDataset(args.labels, max_rows=args.max)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, collate_fn=collate,
    )

    net = GungnirHalfKA().to(device)
    if args.warm_start:
        print(f"Warm-starting from {args.warm_start}", flush=True)
        state = torch.load(args.warm_start, map_location=device, weights_only=True)
        net.load_state_dict(state)
    opt = optim.Adam(net.parameters(), lr=args.lr)

    # Sigmoid-MSE loss tuned so training equilibrium places `pred` in
    # SF-internal-cp (208 per pawn), which is exactly the scale eval.cpp's
    # `raw * 100 / 208` expects for Gungnir cp output.
    #
    #   engine: return pred * 100 / 208   (want = label in pawn-cp)
    #     => pred must converge to label * 2.08
    #   trainer: pred_sig = sigmoid(pred / 416), target = sigmoid(score / 200)
    #     equilibrium: pred / 416 = score / 200  =>  pred = score * 2.08  ✓
    #
    # The 200/416 pair keeps |score|<400cp firmly in the sigmoid's linear
    # region (strong gradient for typical middlegame positions) and rolls
    # off gently toward saturation around |score|>=1500 cp. Mate-labelled
    # positions (±30000) saturate cleanly — gradient decays naturally, no
    # clipping needed.
    PRED_SIG_DIV   = 416.0
    TARGET_SIG_DIV = 200.0
    def compute_loss(pred, score):
        pred_sig = torch.sigmoid(pred / PRED_SIG_DIV)
        target   = torch.sigmoid(score / TARGET_SIG_DIV)
        return ((pred_sig - target) ** 2).mean()

    print(f"Training {args.epochs} epoch(s), batch={args.batch}, lr={args.lr}, samples={len(ds)}", flush=True)
    for epoch in range(args.epochs):
        net.train()
        t0 = time.time()
        total_loss, n_batches = 0.0, 0
        for batch in loader:
            wf, wo, bf, bo, stm, bk, sc = [t.to(device) for t in batch]
            pred = net(wf, wo, bf, bo, stm, bk)
            loss = compute_loss(pred, sc)
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
