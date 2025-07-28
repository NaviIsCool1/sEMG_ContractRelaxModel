#!/usr/bin/env python3
"""
create_splits.py

– Scans processed_npz/ for all .npz files (healthy + stroke windows).
– Shuffles them with a fixed seed for reproducibility.
– Splits into train (70%), val (15%), and test (15%).
– Writes three manifests: train.txt, val.txt, test.txt,
  each line pointing to one .npz path.

These splits ensure your Conv1D+LSTM model sees a balanced
mix of healthy and stroke data at training time, which is
critical for out-of-the-box generalization.
"""

import os
import glob
import random

# SETTINGS
PROCESSED_DIR   = "processed_npz"
OUTPUT_MANIFEST = ["train.txt", "val.txt", "test.txt"]
SEED            = 42
SPLITS          = (0.70, 0.15, 0.15)  # train, val, test

def main():
    # 1) gather
    all_files = glob.glob(os.path.join(PROCESSED_DIR, "*.npz"))
    if not all_files:
        raise RuntimeError(f"No .npz files found in {PROCESSED_DIR}")

    # 2) shuffle
    random.seed(SEED)
    random.shuffle(all_files)

    # 3) split indices
    n = len(all_files)
    n_train = int(n * SPLITS[0])
    n_val   = int(n * SPLITS[1])
    train   = all_files[:n_train]
    val     = all_files[n_train:n_train + n_val]
    test    = all_files[n_train + n_val:]

    # 4) write manifests
    for manifest, subset in zip(OUTPUT_MANIFEST, (train, val, test)):
        with open(manifest, "w") as f:
            for path in subset:
                f.write(path + "\n")
        print(f"Wrote {len(subset)} entries to {manifest}")

if __name__ == "__main__":
    main()
