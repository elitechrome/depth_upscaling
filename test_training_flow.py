
import torch
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from depth_upscaling.data.loader import get_loader
from depth_upscaling.model.network import SparseToDenseNet

def test_full_pipeline():
    print("Step 1: Initializing Data Loader...")
    dataset_url = "dataset_shards/shard-000000.tar"
    if not os.path.exists(dataset_url):
        print(f"Error: {dataset_url} not found. Please run writer.py first.")
        return
        
    loader = get_loader(dataset_url, batch_size=4)
    
    print("Step 2: Initializing SparseToDenseNet Model...")
    model = SparseToDenseNet()
    model.eval()
    
    print("Step 3: Fetching Batch and Running Forward Pass...")
    # Iterators for WebDataset return batches as tuples of lists/tensors
    # Since we used .batched(4), each element in the return is a batch
    
    try:
        for i, batch in enumerate(loader):
            # Unpack batch
            # rgb: (B, C, H, W)
            # depth_gt: (B, 1, H, W)
            # depth_sparse: (B, 1, H_s, W_s)
            
            # WDS .batched returns lists for some types, or tensors if map returned tensors
            rgb, gt, sparse, meta = batch
            
            # WebDataset .batched with tensors usually handles stacking automatically 
            # if the transform returns torch tensors.
            
            # Ensure they are tensors
            if isinstance(rgb, list):
                rgb = torch.stack(rgb)
            if isinstance(sparse, list):
                sparse = torch.stack(sparse)
            
            print(f"Input RGB Batch Shape: {rgb.shape}")
            print(f"Input Sparse Depth Batch Shape: {sparse.shape}")
            
            with torch.no_grad():
                output = model(rgb, sparse)
                
            print(f"Model Output Shape: {output.shape}")
            print("Successfully completed one batch!")
            break
    except Exception as e:
        print(f"Pipeline Test Failed: {e}")

if __name__ == "__main__":
    test_full_pipeline()
