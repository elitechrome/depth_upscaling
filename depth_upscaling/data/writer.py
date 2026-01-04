import webdataset as wds
import os
import numpy as np
import io
import json
import argparse
from depth_upscaling.data.generator import SyntheticGenerator

def write_dataset(output_path, count=100, shards=1):
    """
    Generates 'count' samples and writes them to WebDataset shards.
    """
    os.makedirs(output_path, exist_ok=True)
    pattern = os.path.join(output_path, "shard-%06d.tar")
    
    gen = SyntheticGenerator()
    
    with wds.ShardWriter(pattern, maxcount=count//shards) as sink:
        for i in range(count):
            data = gen.create_mock_scene_buffers()
            
            # Encode images
            # RGB -> jpg
            # Depth -> png (16bit or just float32 bytes for simplicity in python)
            # For this demo, we'll settle for .npy for high precision depth, or .pyd for python pickup
            # Standard WDS: .jpg for RGB, .png for Depth? 
            # Depth 16u: (depth * 1000).astype(uint16) for millimeters
            
            # Encode images using PIL
            from PIL import Image
            
            # RGB
            # numpy array (H,W,3) -> PIL
            rgb_img = Image.fromarray(data["rgb"])
            rgb_bytes = io.BytesIO()
            rgb_img.save(rgb_bytes, format="JPEG", quality=95)
            
            # Depth GT
            # Float meters -> Millimeters uint16
            depth_gt_mm = (data["depth_gt"] * 1000).astype(np.uint16) # (H,W)
            gt_img = Image.fromarray(depth_gt_mm, mode="I;16")
            gt_bytes = io.BytesIO()
            gt_img.save(gt_bytes, format="PNG")
            
            # Depth Sparse
            # Same treatment
            depth_sparse_mm = (data["depth_sparse"] * 1000).astype(np.uint16)
            sparse_img = Image.fromarray(depth_sparse_mm, mode="I;16")
            sparse_bytes = io.BytesIO()
            sparse_img.save(sparse_bytes, format="PNG")
            
            # Meta
            meta = {
                "id": i,
                "rgb_shape": data["rgb"].shape,
                "depth_shape": data["depth_gt"].shape,
                "sparse_shape": data["depth_sparse"].shape
            }
            
            sample = {
                "__key__": f"{i:06d}",
                "rgb.jpg": rgb_bytes.getvalue(),
                "depth_gt.png": gt_bytes.getvalue(),
                "depth_sparse.png": sparse_bytes.getvalue(),
                "json": meta
            }
            sink.write(sample)
    
    print(f"Dataset written to {output_path} with {count} samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./dataset_shards")
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()
    
    write_dataset(args.output, args.count)
