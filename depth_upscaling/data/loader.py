
import webdataset as wds
import torch
from itertools import islice

def get_loader(url, batch_size=4, num_workers=2):
    """
    Returns a PyTorch-like iterator for the dataset.
    """
    # Decoding: 
    # RGB ("jpg") -> 8-bit, [0, 255]
    # Depth ("png") -> 16-bit uint, [0, 65535] (mm)
    
    dataset = (
        wds.WebDataset(url)
        .shuffle(100)
        .decode() # Decode to raw bytes
        .to_tuple("rgb.jpg", "depth_gt.png", "depth_sparse.png", "json")
    )
    
    def transform(sample):
        import numpy as np
        from PIL import Image
        import io
        rgb_bytes, gt_bytes, sparse_bytes, meta = sample
        
        # Decode RGB
        rgb_pil = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
        rgb = torch.from_numpy(np.array(rgb_pil).transpose(2, 0, 1)).float() / 255.0
        
        # Decode Depth (16-bit)
        gt_pil = Image.open(io.BytesIO(gt_bytes)) # Should be I;16
        depth_gt = torch.from_numpy(np.array(gt_pil).astype(np.float32)).unsqueeze(0) / 1000.0
        
        sparse_pil = Image.open(io.BytesIO(sparse_bytes))
        depth_sparse = torch.from_numpy(np.array(sparse_pil).astype(np.float32)).unsqueeze(0) / 1000.0
        
        return rgb, depth_gt, depth_sparse, meta

    mapped_dataset = dataset.map(transform)
    
    # Batching manually since no torch DataLoader
    # wds has .batched(batch_size)
    batched_dataset = mapped_dataset.batched(batch_size)
    
    return batched_dataset

if __name__ == "__main__":
    # Test
    url = "./dataset_shards/shard-000000.tar"
    loader = get_loader(url)
    
    for i, batch in enumerate(loader):
        rgb, gt, sparse, meta = batch
        print(f"Batch {i}: RGB {rgb.shape}, GT {gt.shape}, Sparse {sparse.shape}")
        if i >= 2: break
