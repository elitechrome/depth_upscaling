import numpy as np
import os
import io

# Optional: Try importing open3d or trimesh if available, else mock
try:
    import open3d as o3d
except ImportError:
    o3d = None

from depth_upscaling.data.simulation import SensorConfig, DepthSimulator

class SyntheticGenerator:
    def __init__(self):
        self.sim = DepthSimulator()
        
    def create_mock_scene_buffers(self):
        """
        Creates a dummy scene with analytical geometry (e.g. a sphere or plane)
        to verify the pipeline without needing 3D rendering libs yet.
        """
        # RGB: 800x600
        # Create a gradient for RGB
        x = np.linspace(0, 1, 800)
        y = np.linspace(0, 1, 600)
        xx, yy = np.meshgrid(x, y)
        rgb = np.stack([xx, yy, np.zeros_like(xx)], axis=-1) * 255
        rgb = rgb.astype(np.uint8)
        
        # GT Depth: 800x600
        # Let's make a slanted plane: z = 1.0 + x*1.0 + y*0.5
        # In camera frame, x,y are normalized coordinates.
        params = self.sim.rgb_sensor.get_pixel_rays() # [600, 800, 3]
        # rays are direction vectors.
        # Intersect with plane Z=2
        # Ray: P = t * D
        # Z = t * Dz = 2 => t = 2 / Dz
        # D is standard vector (x,y,z=1). 
        # Actually existing get_pixel_rays logic returns normalized vectors.
        # But for depth map generation we often use "axis-aligned depth" (Z value)
        # or "Euclidean distance". Standard is Z-depth.
        
        # Simple Mock:
        # Just filling depth with a gradient 1.0m -> 5.0m
        depth_gt = 1.0 + (xx * 4.0).astype(np.float32)
        
        # Simulate Sparse Depth Input
        # We need to sampling from this GT depth at the specific sparse points.
        # BUT: The sparse sensor is 224x100 and at a different FOV.
        # Step 1: Create dense depth map for DEPTH SENSOR View.
        # If Extrinsics are Identity, we can just resample the scene for the new FOV.
        
        # Depth Sensor Rays
        d_rays = self.sim.depth_sensor.get_pixel_rays()
        # Similar plane intersection: z=2 (mock)
        # For simplicity in mock, let's just use the same gradient logic but mapped to new FOV?
        # No, let's just make a simple pattern.
        
        # Re-using simulation logic:
        # We need "True" depth at the Depth Sensor pixels.
        # Let's assume the scene is a wall at Z=2.0m for everything.
        depth_sensor_view = np.ones((100, 224), dtype=np.float32) * 2.0
        
        # Apply mask
        mask = self.sim.generate_sparse_pattern_mask(angular_spacing_deg=1.5)
        
        # Sparse Depth: Valid values where mask is True, else 0
        depth_sparse = depth_sensor_view * mask
        
        return {
            "rgb": rgb,
            "depth_gt": depth_gt,
            "depth_sparse": depth_sparse
        }

if __name__ == "__main__":
    gen = SyntheticGenerator()
    data = gen.create_mock_scene_buffers()
    print("Generated Keys:", data.keys())
    print("RGB shape:", data["rgb"].shape)
    print("Sparse Depth valid pixels:", np.count_nonzero(data["depth_sparse"]))
