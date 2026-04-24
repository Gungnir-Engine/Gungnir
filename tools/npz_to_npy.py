"""
Convert a .npz (zip of .npy files) to individual .npy files in a directory.
Enables true mmap_mode='r' on each array.
"""
import sys
import os
import numpy as np

src = sys.argv[1]
out_dir = sys.argv[2]
os.makedirs(out_dir, exist_ok=True)

print(f"Loading {src}...", flush=True)
data = np.load(src, allow_pickle=False)
for key in data.files:
    arr = data[key]
    out_path = os.path.join(out_dir, f'{key}.npy')
    print(f"  {key}: shape={arr.shape} dtype={arr.dtype} -> {out_path}", flush=True)
    np.save(out_path, arr)
print("Done.")
