"""Check split_info.json for data leakage"""
import json, os

data_dir = "/home/cristian/geotermia/data/processed"
with open(os.path.join(data_dir, "split_info.json")) as f:
    info = json.load(f)

for k, v in info.items():
    if isinstance(v, (str, int, float, bool)):
        print(f"{k}: {v}")
    elif isinstance(v, list) and len(v) < 10:
        print(f"{k}: {v}")
    elif isinstance(v, list):
        print(f"{k}: list[{len(v)}]  first3={v[:3]}")
    elif isinstance(v, dict):
        print(f"{k}: dict keys={list(v.keys())[:10]}")

# Check for group overlap
if "groups_train" in info and "groups_val" in info:
    gt = set(info["groups_train"])
    gv = set(info["groups_val"])
    gt2 = set(info.get("groups_test", []))
    print(f"\nTrain groups: {len(gt)}")
    print(f"Val groups: {len(gv)}")
    print(f"Test groups: {len(gt2)}")
    overlap_tv = gt & gv
    overlap_tt = gt & gt2
    print(f"Overlap train&val: {len(overlap_tv)}")
    print(f"Overlap train&test: {len(overlap_tt)}")
    if overlap_tv:
        print(f"Leaked groups (train&val): {sorted(overlap_tv)[:10]}")

# Check filenames
for split in ["train", "val", "test"]:
    key = f"{split}_files"
    if key in info:
        files = info[key]
        print(f"\n{split}: {len(files)} files")
        # Extract base names (remove aug suffixes)
        bases = set()
        for f in files:
            stem = os.path.splitext(f)[0]
            for s in ["_original", "_rotation_90", "_rotation_180", "_rotation_270",
                       "_flip_horizontal", "_flip_vertical", "_brightness_1.2"]:
                if stem.endswith(s):
                    stem = stem[:-len(s)]
                    break
            bases.add(stem)
        print(f"  Unique base images: {len(bases)}")

# Check if filenames stored differently
if "train_filenames" in info or "filenames" in info:
    print("\nAlternative filename keys found")
