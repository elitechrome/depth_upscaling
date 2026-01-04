
import os
import glob
import argparse
import numpy as np
import cv2
from PIL import Image
import webdataset as wds
import io
# Adjust path for local verification if needed
import sys
sys.path.append(os.getcwd())

from depth_upscaling.data.simulation import DepthSimulator

def process_arkit_scenes(input_root, output_dir, limit=None, split="Training"):
    """
    input_root: Path to ARKitScenes/depth_upsampling
    output_dir: Path to save shards
    limit: Max samples (default None)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # We look for highres_depth files, as they are the GT.
    # Pattern: input_root/Training/*/highres_depth/*.png
    search_path = os.path.join(input_root, split, "*", "highres_depth", "*.png")
    depth_files = sorted(glob.glob(search_path))
    
    if limit:
        depth_files = depth_files[:limit]
        
    print(f"Found {len(depth_files)} samples in {split}")
    
    sim = DepthSimulator()
    
    # Pre-calculate center crop box for 4:3 aspect ratio from 1920x1440 (which is 4:3!)
    # Actually 1920/1440 = 1.333. 
    # Target 800x600 = 1.333.
    # So we can direct resize or crop-resize. 
    # ARKitScenes RGB is 1920x1440.
    # Depth HighRes is 1920x1440.
    # Perfect match. We just resize.
    
    pattern = os.path.join(output_dir, f"{split.lower()}-shard-%06d.tar")
    
    # Shard writing
    with wds.ShardWriter(pattern, maxcount=500) as sink:
        for idx, depth_path in enumerate(depth_files):
            # Parse paths
            # .../Training/<vid>/highres_depth/<frame>.png
            # Need Color: .../Training/<vid>/color/<frame>.png
            
            parent = os.path.dirname(depth_path) # highres_depth
            grandparent = os.path.dirname(parent) # vid
            
            fname = os.path.basename(depth_path)
            color_path = os.path.join(grandparent, "color", fname)
            
            if not os.path.exists(color_path):
                print(f"Missing color for {fname}, skipping.")
                continue
                
            # Load Data
            # RGB
            bgr = cv2.imread(color_path)
            if bgr is None: continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) # 1440x1920? Check shape.
            
            # Depth GT (mm unit usually in ARKitScenes, let's verify doc? "projected from mesh")
            # Usually png 16bit.
            depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) # uint16 mm
            if depth_raw is None: continue
            
            # Resize to 800x600
            # cv2.resize expects (width, height)
            rgb_800 = cv2.resize(rgb, (800, 600), interpolation=cv2.INTER_AREA)
            depth_800_mm = cv2.resize(depth_raw, (800, 600), interpolation=cv2.INTER_NEAREST)
            
            # Convert mm to meters for simulation
            depth_800_m = depth_800_mm.astype(np.float32) / 1000.0
            
            # Simulate Sparse
            # Returns (100, 224) array with meters, zeros where invalid
            sparse_depth_m = sim.simulate_from_dense_depth(depth_800_m)
            
            # Encode for WDS
            # 1. RGB -> JPG
            rgb_pil = Image.fromarray(rgb_800)
            rgb_bytes = io.BytesIO()
            rgb_pil.save(rgb_bytes, format="JPEG", quality=95)
            
            # 2. Depth GT -> PNG 16bit
            depth_gt_pil = Image.fromarray(depth_800_mm, mode="I;16")
            gt_bytes = io.BytesIO()
            depth_gt_pil.save(gt_bytes, format="PNG")
            
            # 3. Sparse Depth -> PNG 16bit (mm)
            sparse_mm = (sparse_depth_m * 1000).astype(np.uint16)
            sparse_pil = Image.fromarray(sparse_mm, mode="I;16")
            sparse_bytes = io.BytesIO()
            sparse_pil.save(sparse_bytes, format="PNG")
            
            meta = {
                "original_file": depth_path,
                "dataset": "ARKitScenes",
                "split": split
            }
            
            sample = {
                "__key__": f"{split}_{idx:06d}",
                "rgb.jpg": rgb_bytes.getvalue(),
                "depth_gt.png": gt_bytes.getvalue(),
                "depth_sparse.png": sparse_bytes.getvalue(),
                "json": meta
            }
            sink.write(sample)
            
            if idx % 100 == 0:
                print(f"Processed {idx} samples...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Root of ARKitScenes/depth_upsampling")
    parser.add_argument("--output", default="./arkit_shards")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", default="Training")
    
    args = parser.parse_args()
    process_arkit_scenes(args.input, args.output, args.limit, args.split)
