import sys
import os
import torch

# Add the cloned repo to path so we can import src
repo_path = os.path.join(os.getcwd(), 'depth_upscaling/depthor_repo')
sys.path.append(repo_path)

try:
    from src.models.depthor import Depthor
except ImportError as e:
    print(f"Failed to import Depthor: {e}")
    sys.exit(1)

def verify_depthor():
    print("Initializing Depthor Model...")
    # n_bins=100 is standard
    try:
        model = Depthor(n_bins=100)
        model.set_extra_param('cpu') # Initialize mean/std buffers
    except Exception as e:
        print(f"Initialization Failed: {e}")
        # Sometimes 'vits' loading relies on internet or cached weights.
        # depthor.py: self.depth_anything = set_depthanything(encoder='vits')
        # This might fail if weights aren't downloaded.
        return

    print("Model Initialized. Running Forward Pass...")
    
    # Input dict as per Depthor.forward
    # x = input_data['image']  # [b, 3, 480, 640]
    # sparse_depth = input_data['sparse_depth']  # [b, 1, 480, 640]
    
    B, C, H, W = 1, 3, 480, 640
    dummy_img = torch.randn(B, C, H, W)
    dummy_sparse = torch.randn(B, 1, H, W)
    
    input_data = {
        'image': dummy_img,
        'sparse_depth': dummy_sparse
    }
    
    # Mocking register_buffer issue if devices don't match?
    # Depthor uses 'vits' and automatically loads it. 
    # Let's hope it runs on CPU by default.
    
    try:
        depth_0, final = model(input_data)
        print(f"Success! Output Shapes: Coarse {depth_0.shape}, Final {final.shape}")
    except Exception as e:
        print(f"Forward Pass Failed: {e}")

if __name__ == "__main__":
    verify_depthor()
