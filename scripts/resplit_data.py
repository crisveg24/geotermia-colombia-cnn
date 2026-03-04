#!/usr/bin/env python3
# Copyright (c) 2025-2026 Vega Sánchez · Arévalo Rubiano · Espitia Ayala · Rivera Martín
# Universidad de San Buenaventura — Bogotá | github.com/crisveg24/geotermia-colombia-cnn
"""
resplit_data.py - Re-split processed .npy data fixing data leakage.

Strategy (memory-efficient for 15GB RAM):
1. Read split_info.json to get filenames for ALL samples
2. Compute new group-based splits (just on filenames, no pixel data)
3. For each new split, iterate over OLD .npy parts via memmap,
   extract the samples that belong to this new split, write new .npy parts

This avoids loading all 30GB into RAM at once.
"""
import json
import sys
import gc
import numpy as np
from pathlib import Path
from collections import defaultdict

# --- CONFIG ---
BASE_DIR = Path("/home/cristian/geotermia/data/processed")
OUTPUT_DIR = Path("/home/cristian/geotermia/data/processed_v2")
BATCH_SAVE = 500
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42
DRY_RUN = "--dry-run" in sys.argv


def extract_group(filename):
    """
    Extract geographic zone from filename, stripping augmentation
    and grid suffixes. All tiles + augmentations from the same
    geographic zone get the same group.
    
    Examples:
      Cerro_Bravo_center_rotation_90.tif -> Cerro_Bravo
      CL_Apacheta_E_original.tif -> CL_Apacheta
      Amazonas_Leticia_brightness_1.2.tif -> Amazonas_Leticia
      CL_Ancud_Ctrl_NW_flip_horizontal.tif -> CL_Ancud_Ctrl
    """
    name = Path(filename).stem

    # Step 1: Strip augmentation suffix
    aug_suffixes = [
        '_rotation_neg45', '_rotation_45', '_rotation_90',
        '_rotation_180', '_rotation_270',
        '_flip_horizontal', '_flip_vertical',
        '_brightness_1.2', '_brightness_0.8',
        '_contrast_1.3', '_contrast_0.7',
        '_noise_0.02', '_noise_small', '_noise_medium',
        '_blur_light', '_blur_medium',
        '_crop_0.9', '_crop_0.85',
        '_rot90_flip_h', '_rot180_bright',
        '_flip_v_contrast', '_rot45_noise', '_crop_blur', '_bright_blur',
        '_contrast_noise', '_rot90_crop', '_rot180_contrast', '_flip_h_bright',
        '_rot270_blur', '_crop_contrast_noise', '_rot45_bright_blur',
        '_original',
    ]
    for suf in sorted(aug_suffixes, key=len, reverse=True):
        if name.endswith(suf):
            name = name[:-len(suf)]
            break

    # Step 2: Strip grid suffix
    grid_suffixes = [
        '_northeast', '_northwest', '_southeast', '_southwest',
        '_center', '_north', '_south', '_east', '_west',
        '_NE', '_NW', '_SE', '_SW',
        '_N', '_S', '_E', '_W',
        '_norte', '_sur',
    ]
    for suf in sorted(grid_suffixes, key=len, reverse=True):
        if name.endswith(suf):
            name = name[:-len(suf)]
            break

    return name


def build_global_index():
    """
    Build a global index mapping each sample to its location in old .npy files.
    Does NOT load pixel data.
    """
    with open(BASE_DIR / "split_info.json") as f:
        info = json.load(f)

    entries = []
    for split in ["train", "val", "test"]:
        filenames = info[f"{split}_files"]
        labels = np.load(BASE_DIR / f"y_{split}.npy")
        assert len(filenames) == len(labels)

        for i, (fn, lbl) in enumerate(zip(filenames, labels)):
            entries.append({
                "filename": fn,
                "label": int(lbl),
                "old_split": split,
                "old_idx": i,
            })

    print(f"Total samples: {len(entries)}")
    return entries, info


def compute_new_splits(entries):
    """Compute new train/val/test indices with zero leakage."""
    from sklearn.model_selection import GroupShuffleSplit

    n = len(entries)
    filenames = [e["filename"] for e in entries]
    labels = np.array([e["label"] for e in entries], dtype=np.int32)
    groups = np.array([extract_group(f) for f in filenames])

    unique_groups = np.unique(groups)
    print(f"Unique geographic zones (groups): {len(unique_groups)}")

    g_counts = defaultdict(int)
    for g in groups:
        g_counts[g] += 1
    sizes = sorted(g_counts.values(), reverse=True)
    print(f"Samples per group: max={sizes[0]}, min={sizes[-1]}, "
          f"median={sizes[len(sizes)//2]}, mean={np.mean(sizes):.1f}")

    indices = np.arange(n)

    gss_test = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    temp_idx, test_idx = next(gss_test.split(indices, labels, groups))

    val_adjusted = VAL_SIZE / (1 - TEST_SIZE)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_adjusted, random_state=RANDOM_STATE)
    train_sub, val_sub = next(gss_val.split(
        indices[temp_idx], labels[temp_idx], groups[temp_idx]
    ))
    train_idx = temp_idx[train_sub]
    val_idx = temp_idx[val_sub]

    # Verify zero leakage
    g_train = set(groups[train_idx])
    g_val = set(groups[val_idx])
    g_test = set(groups[test_idx])

    leak_tv = g_train & g_val
    leak_tt = g_train & g_test
    leak_vt = g_val & g_test

    print(f"\n{'='*50}")
    print(f"NEW SPLIT RESULTS")
    print(f"{'='*50}")
    print(f"Train: {len(train_idx)} samples, {len(g_train)} zones")
    print(f"Val:   {len(val_idx)} samples, {len(g_val)} zones")
    print(f"Test:  {len(test_idx)} samples, {len(g_test)} zones")

    y_tr = labels[train_idx]
    y_va = labels[val_idx]
    y_te = labels[test_idx]
    print(f"\nTrain: pos={y_tr.sum()}, neg={(1-y_tr).sum()}, ratio={y_tr.mean():.3f}")
    print(f"Val:   pos={y_va.sum()}, neg={(1-y_va).sum()}, ratio={y_va.mean():.3f}")
    print(f"Test:  pos={y_te.sum()}, neg={(1-y_te).sum()}, ratio={y_te.mean():.3f}")

    print(f"\nLeakage check:")
    print(f"  Train-Val:  {len(leak_tv)} groups")
    print(f"  Train-Test: {len(leak_tt)} groups")
    print(f"  Val-Test:   {len(leak_vt)} groups")

    if leak_tv or leak_tt or leak_vt:
        print("  LEAKAGE DETECTED! Aborting.")
        sys.exit(1)
    else:
        print("  ZERO leakage - all clear!")

    return train_idx, val_idx, test_idx, labels


def write_new_split(entries, global_indices, new_split_name):
    """
    Write new .npy files using memmap to read old parts.
    Groups requests by old part file to minimize I/O.
    """
    n = len(global_indices)
    print(f"\nWriting {new_split_name}: {n} samples...")

    # Save labels
    labels = np.array([entries[gi]["label"] for gi in global_indices], dtype=np.int32)
    filenames = [entries[gi]["filename"] for gi in global_indices]
    np.save(OUTPUT_DIR / f"y_{new_split_name}.npy", labels)

    # Process in output batches of BATCH_SAVE
    n_out_parts = 0
    for out_start in range(0, n, BATCH_SAVE):
        out_end = min(out_start + BATCH_SAVE, n)
        batch_size = out_end - out_start

        # Figure out where each sample lives in old files
        batch_sources = []
        for new_pos in range(out_start, out_end):
            gi = global_indices[new_pos]
            e = entries[gi]
            part_idx = e["old_idx"] // BATCH_SAVE
            local_idx = e["old_idx"] % BATCH_SAVE
            batch_sources.append((e["old_split"], part_idx, local_idx))

        # Group by (old_split, part_idx) to minimize file opens
        needed_parts = defaultdict(list)
        for i, (os_, pi, li) in enumerate(batch_sources):
            needed_parts[(os_, pi)].append((i, li))

        # Allocate output batch
        out_batch = np.empty((batch_size, 224, 224, 7), dtype=np.float32)

        # Load each needed old part via memmap, extract samples
        for (os_, pi), requests in needed_parts.items():
            part_path = BASE_DIR / f"X_{os_}_part{pi}.npy"
            data = np.load(part_path, mmap_mode='r')
            for out_i, local_i in requests:
                out_batch[out_i] = data[local_i]
            del data

        np.save(OUTPUT_DIR / f"X_{new_split_name}_part{n_out_parts}.npy", out_batch)
        print(f"  Saved X_{new_split_name}_part{n_out_parts}.npy: {out_batch.shape}")
        n_out_parts += 1
        del out_batch
        gc.collect()

    return filenames, n_out_parts


def main():
    print("=" * 60)
    print("RESPLIT DATA - Fixing data leakage (memory-efficient)")
    print("=" * 60)

    if DRY_RUN:
        print("\n*** DRY RUN - no files will be written ***\n")

    # Step 1: Build global index (no pixel data)
    print("\n[1/4] Building global index...")
    entries, info = build_global_index()

    # Step 2: Compute new splits
    print("\n[2/4] Computing new splits...")
    train_idx, val_idx, test_idx, labels = compute_new_splits(entries)

    if DRY_RUN:
        print("\nDry run complete.")
        return

    # Step 3: Create output directory
    print(f"\n[3/4] Creating output: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 4: Write new split files
    print("\n[4/4] Writing new .npy files...")
    split_info = {"split_parts": {}}

    for split_name, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        filenames, n_parts = write_new_split(entries, indices, split_name)
        split_info[f"{split_name}_files"] = filenames
        split_info["split_parts"][split_name] = n_parts

    # Copy band_stats.json
    import shutil
    shutil.copy2(BASE_DIR / "band_stats.json", OUTPUT_DIR / "band_stats.json")

    # Metadata
    split_info["target_size"] = [224, 224]
    split_info["num_bands"] = 7

    # Class weights
    y_train = labels[train_idx]
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    total = n_pos + n_neg
    split_info["class_weights"] = {
        "0": round(total / (2.0 * n_neg), 4),
        "1": round(total / (2.0 * n_pos), 4)
    }

    with open(OUTPUT_DIR / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"DONE! New data saved to: {OUTPUT_DIR}")
    for f in sorted(OUTPUT_DIR.iterdir()):
        size_mb = f.stat().st_size / (1024*1024)
        print(f"  {f.name}: {size_mb:.1f} MB")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
