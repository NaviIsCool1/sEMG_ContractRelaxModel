import os, glob

def collect_files(base_dir):
    pairs = []
    for cls in ('contract','relax','onset','offset'):
        sig_dir = os.path.join(base_dir, f"myo_{cls}4")
        lbl_dir = os.path.join(base_dir, f"myo_{cls}_labels4")
        # e.g. '/…/myo_contract4/contract_1.csv'
        for sig_path in glob.glob(os.path.join(sig_dir, f"{cls}_*.csv")):
            fn = os.path.basename(sig_path)            # 'contract_1.csv'
            idx = fn.split('_')[1].split('.')[0]       # '1'
            lbl_name = f"labels_{cls}_{idx}.csv"
            lbl_path = os.path.join(lbl_dir, lbl_name)
            if os.path.isfile(lbl_path):
                pairs.append((sig_path, lbl_path))
            else:
                print(f"Missing label for {sig_path}: expected {lbl_path}")
    return pairs

# Example usage:
stroke_pairs  = collect_files("/home/navneeth/sEMG_ContractRelaxModel/Stroke Data")
healthy_pairs = collect_files("/home/navneeth/sEMG_ContractRelaxModel/Healthy Data")

print(f"Found {len(stroke_pairs)} stroke files and {len(healthy_pairs)} healthy files.")
