# prepare_npz.py

"""
prepare_npz.py

This script converts every raw-CSV window and its matching label-CSV
into a compact `.npz` archive, placing them into a `processed_npz/` directory.

It now **ignores** any signal/label pairs whose lengths don't match
(or aren't exactly 2000 samples), printing a warning and continuing.

Usage:
  (venv) $ python prepare_npz.py
"""

import os
import glob
import numpy as np

COHORTS = ["Healthy Data", "Stroke Data"]
CLASSES = ["contract", "relax", "onset", "offset"]
OUT_DIR = "processed_npz"
EXPECTED_LEN = 2000   # expected samples per window

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    saved = 0
    skipped = 0

    for cohort in COHORTS:
        for cls in CLASSES:
            sig_dir = os.path.join(cohort, f"myo_{cls}4")
            lbl_dir = os.path.join(cohort, f"myo_{cls}_labels4")
            for sig_path in glob.glob(os.path.join(sig_dir, f"{cls}_*.csv")):
                idx = os.path.basename(sig_path).split("_")[1].split(".")[0]
                lbl_path = os.path.join(lbl_dir, f"labels_{cls}_{idx}.csv")

                # Load data
                sig = np.loadtxt(sig_path, delimiter=",")
                lbl = np.loadtxt(lbl_path, delimiter=",")

                # Check lengths
                if sig.shape[0] != lbl.shape[0] or sig.shape[0] != EXPECTED_LEN:
                    print(f"⚠️ Skipping mismatch or invalid length:")
                    print(f"    {sig_path}: {sig.shape[0]} samples")
                    print(f"    {lbl_path}: {lbl.shape[0]} samples")
                    skipped += 1
                    continue

                # Save .npz
                out_name = f"{cohort.replace(' ', '_')}_{cls}_{idx}.npz"
                out_path = os.path.join(OUT_DIR, out_name)
                np.savez(out_path, signal=sig, label=lbl)
                saved += 1

    print(f"✅ Saved {saved} files into '{OUT_DIR}/'")
    print(f"⚠️ Skipped {skipped} mismatched or invalid windows")

if __name__ == "__main__":
    main()

