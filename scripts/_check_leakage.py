"""Deep check: verify group extraction and find leakage"""
import json, os, re
from collections import Counter

data_dir = "/home/cristian/geotermia/data/processed"
with open(os.path.join(data_dir, "split_info.json")) as f:
    info = json.load(f)

# Extract groups from filenames using the SAME logic as prepare_dataset.py
aug_suffixes = [
    '_original', '_rotation_90', '_rotation_180', '_rotation_270',
    '_rotation_45', '_rotation_neg45', '_flip_horizontal', '_flip_vertical',
    '_brightness_1.2', '_brightness_0.8', '_contrast_1.3', '_contrast_0.7',
    '_noise_small', '_noise_medium', '_blur_light', '_blur_medium',
    '_crop_0.9', '_crop_0.85', '_rot90_flip_h', '_rot180_bright',
    '_flip_v_contrast', '_rot45_noise', '_crop_blur', '_bright_blur',
    '_contrast_noise', '_rot90_crop', '_rot180_contrast', '_flip_h_bright',
    '_rot270_blur', '_crop_contrast_noise', '_rot45_bright_blur',
]

grid_suffixes = ['_center', '_N', '_S', '_E', '_W', '_NE', '_NW', '_SE', '_SW']

def extract_group(filename):
    stem = os.path.splitext(filename)[0]
    group = stem
    for suffix in aug_suffixes:
        if group.endswith(suffix):
            group = group[:-len(suffix)]
            break
    for suffix in grid_suffixes:
        if group.endswith(suffix):
            group = group[:-len(suffix)]
            break
    return group

# Extract groups for each split
groups = {}
for split in ["train", "val", "test"]:
    key = f"{split}_files"
    files = info[key]
    split_groups = [extract_group(f) for f in files]
    groups[split] = set(split_groups)
    print(f"{split}: {len(files)} files, {len(groups[split])} unique groups")

# Check overlap
overlap_tv = groups["train"] & groups["val"]
overlap_tt = groups["train"] & groups["test"]
overlap_vt = groups["val"] & groups["test"]
print(f"\nOverlap train&val: {len(overlap_tv)} groups")
print(f"Overlap train&test: {len(overlap_tt)} groups")
print(f"Overlap val&test: {len(overlap_vt)} groups")

if overlap_tv:
    print(f"\nSample leaked groups (train&val): {sorted(overlap_tv)[:20]}")

# Check specific case: Cerro_Bravo
print("\n=== Cerro_Bravo analysis ===")
for split in ["train", "val", "test"]:
    key = f"{split}_files"
    cb_files = [f for f in info[key] if f.startswith("Cerro_Bravo")]
    if cb_files:
        cb_groups = set(extract_group(f) for f in cb_files)
        print(f"{split}: {len(cb_files)} Cerro_Bravo files, groups={cb_groups}")
        print(f"  First 5: {cb_files[:5]}")

# Also check: how many ORIGINAL images (before aug) per split  
print("\n=== Original tile distribution ===")
for split in ["train", "val", "test"]:
    key = f"{split}_files"
    tiles = set()
    for f in info[key]:
        stem = os.path.splitext(f)[0]
        for suffix in aug_suffixes:
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                break
        tiles.add(stem)  # This is the tile (zone+grid), NOT the zone
    print(f"{split}: {len(tiles)} unique tiles")
    
    # Now check grid tiles of same zone in this split
    zones = set()
    for t in tiles:
        z = t
        for suffix in grid_suffixes:
            if z.endswith(suffix):
                z = z[:-len(suffix)]
                break
        zones.add(z)
    print(f"  -> {len(zones)} unique zones")

# The key question: are tiles from the SAME zone split across train/val/test?
print("\n=== LEAKAGE ANALYSIS: tiles per zone across splits ===")
all_zone_tiles = {}  # zone -> {split: set of tiles}
for split in ["train", "val", "test"]:
    key = f"{split}_files"
    for f in info[key]:
        stem = os.path.splitext(f)[0]
        for suffix in aug_suffixes:
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                break
        tile = stem
        zone = tile
        for suffix in grid_suffixes:
            if zone.endswith(suffix):
                zone = zone[:-len(suffix)]
                break
        if zone not in all_zone_tiles:
            all_zone_tiles[zone] = {"train": set(), "val": set(), "test": set()}
        all_zone_tiles[zone][split].add(tile)

leaked_zones = 0
clean_zones = 0
for zone, splits_data in all_zone_tiles.items():
    in_splits = [s for s in ["train", "val", "test"] if splits_data[s]]
    if len(in_splits) > 1:
        leaked_zones += 1
    else:
        clean_zones += 1

print(f"Total zones: {len(all_zone_tiles)}")
print(f"Zones in 1 split only (clean): {clean_zones}")
print(f"Zones in multiple splits (LEAKED): {leaked_zones}")
print(f"Leakage rate: {leaked_zones/len(all_zone_tiles)*100:.1f}%")

# Show examples of leaked zones
print("\nExamples of leaked zones:")
count = 0
for zone, splits_data in sorted(all_zone_tiles.items()):
    in_splits = [s for s in ["train", "val", "test"] if splits_data[s]]
    if len(in_splits) > 1 and count < 10:
        print(f"  {zone}:")
        for s in in_splits:
            print(f"    {s}: {sorted(splits_data[s])}")
        count += 1
