"""
train_nnue.py — Distillation training skeleton for Gungnir.

This is a SKELETON / REFERENCE IMPLEMENTATION, not a working SF-quality
trainer. It demonstrates the training loop shape with PyTorch on a
simplified PieceSquare feature set (NOT HalfKAv2_hm).

Usage:
    python train_nnue.py <labels.txt> <output.bin> [--epochs 10] [--lr 1e-3]

Input  (labels.txt): "FEN|score_cp" per line (output of label_fens.py).
Output (output.bin): trained weights as raw float32 (NOT a real .nnue file).

To actually produce a Stockfish-format HalfKAv2_hm .nnue file that Gungnir's
loader can consume, use the official nnue-pytorch repo:
    https://github.com/official-stockfish/nnue-pytorch
…with our labels.txt as input. See tools/README.md for the recipe.

Network architecture (this skeleton):
    Input  : 768 features = 2 colors × 6 piece types × 64 squares
    Layer 1: Linear 768 → 256 + clipped ReLU
    Layer 2: Linear 256 → 32 + clipped ReLU
    Output : Linear 32 → 1 (centipawns from WHITE POV)

Loss: Texel-style sigmoid MSE, target = sigmoid(score / 400). Saturated
positions (|score| > 1500 cp) are dropped.
"""

import argparse
import os
import sys

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    print("ERROR: PyTorch not installed. `pip install torch`.")
    sys.exit(1)


# --- Feature extraction ---------------------------------------------------

PIECE_TO_IDX = {  # (piece, color) -> [0..11]
    ('P', 'w'): 0, ('N', 'w'): 1, ('B', 'w'): 2, ('R', 'w'): 3, ('Q', 'w'): 4, ('K', 'w'): 5,
    ('P', 'b'): 6, ('N', 'b'): 7, ('B', 'b'): 8, ('R', 'b'): 9, ('Q', 'b'):10, ('K', 'b'):11,
}


def fen_to_features(fen: str):
    """Returns a sparse list of indices into the 768-dim input vector."""
    board, *_ = fen.split()
    feats = []
    rank = 7
    for c in board:
        if c == '/':
            rank -= 1
            file = 0
        elif c.isdigit():
            file = (file if 'file' in dir() else 0) + int(c)
        else:
            color = 'w' if c.isupper() else 'b'
            piece = c.upper()
            sq = rank * 8 + (file if 'file' in dir() else 0)
            feats.append(PIECE_TO_IDX[(piece, color)] * 64 + sq)
            file += 1
        if c not in '/' and not c.isdigit():
            pass
    return feats


def features_to_dense(feats, dim=768):
    v = torch.zeros(dim)
    for f in feats:
        v[f] = 1.0
    return v


# --- Model ----------------------------------------------------------------

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.clamp(self.fc1(x), 0.0, 1.0)
        x = torch.clamp(self.fc2(x), 0.0, 1.0)
        return self.fc3(x).squeeze(-1)


# --- Training loop --------------------------------------------------------

def parse_labels(path: str, max_score=1500):
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or '|' not in line:
                continue
            fen, score_str = line.rsplit('|', 1)
            try:
                score = int(score_str)
            except ValueError:
                continue
            if abs(score) > max_score:
                continue   # drop saturated positions
            samples.append((fen, score))
    return samples


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('labels', help='Labels file (FEN|score)')
    ap.add_argument('output', help='Output weights binary')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr',     type=float, default=1e-3)
    ap.add_argument('--batch',  type=int, default=256)
    args = ap.parse_args()

    print(f'Loading labels from {args.labels}...')
    samples = parse_labels(args.labels)
    print(f'  {len(samples)} usable samples (after saturation filter).')
    if not samples:
        sys.exit(1)

    print('Vectorizing features...')
    inputs = torch.zeros(len(samples), 768)
    targets = torch.zeros(len(samples))
    for i, (fen, score) in enumerate(samples):
        feats = fen_to_features(fen)
        for f in feats:
            inputs[i, f] = 1.0
        targets[i] = torch.sigmoid(torch.tensor(score / 400.0))
        if i % 5000 == 0 and i > 0:
            print(f'  {i}/{len(samples)}')

    net = Net()
    opt = optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    n = len(samples)
    print(f'Training {args.epochs} epochs, batch={args.batch}, lr={args.lr}')
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i+args.batch]
            x = inputs[idx]
            y = targets[idx]
            pred = torch.sigmoid(net(x))
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        print(f'  epoch {epoch+1}/{args.epochs}  loss={epoch_loss/n:.6f}')

    print(f'Saving weights to {args.output}...')
    state = net.state_dict()
    blobs = []
    for k, v in state.items():
        blobs.append(v.detach().numpy().astype('float32').tobytes())
    with open(args.output, 'wb') as f:
        for b in blobs:
            f.write(b)
    print('Done.')


if __name__ == '__main__':
    main()
