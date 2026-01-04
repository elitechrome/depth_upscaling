import numpy as np
import torch

class SensorConfig:
    def __init__(self, width, height, hfov_deg, vfov_deg, position_offset=None, rotation_euler=None):
        """
        Args:
            width (int): Pixel resolution width
            height (int): Pixel resolution height
            hfov_deg (float): Horizontal FOV in degrees
            vfov_deg (float): Vertical FOV in degrees
            position_offset (np.array): [x,y,z] offset relative to rig center (meters)
            rotation_euler (np.array): [pitch, yaw, roll] rotation relative to rig (degrees)
        """
        self.width = width
        self.height = height
        self.hfov = np.deg2rad(hfov_deg)
        self.vfov = np.deg2rad(vfov_deg)
        
        # Extrinsics (Rig -> Sensor)
        self.position = np.array(position_offset) if position_offset is not None else np.zeros(3)
        self.rotation = np.array(rotation_euler) if rotation_euler is not None else np.zeros(3)
        
        self.K = self._compute_intrinsics()

    def _compute_intrinsics(self):
        """Computes Pinhole Camera Matrix K."""
        # Focal length from FOV
        # tan(theta/2) = (W/2) / f
        # f = (W/2) / tan(theta/2)
        fx = (self.width / 2.0) / np.tan(self.hfov / 2.0)
        fy = (self.height / 2.0) / np.tan(self.vfov / 2.0)
        cx = self.width / 2.0
        cy = self.height / 2.0
        
        K = np.eye(3)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
        return K

    def get_pixel_rays(self):
        """Returns unit vectors for every pixel in the sensor."""
        x = np.arange(self.width)
        y = np.arange(self.height)
        xx, yy = np.meshgrid(x, y)
        
        # Normalize to camera formulation
        # x_cam = (u - cx) * z / fx
        # y_cam = (v - cy) * z / fy
        # For rays, assume z=1
        x_cam = (xx - self.K[0, 2]) / self.K[0, 0]
        y_cam = (yy - self.K[1, 2]) / self.K[1, 1]
        z_cam = np.ones_like(x_cam)
        
        rays = np.stack([x_cam, y_cam, z_cam], axis=-1)
        # Normalize
        norm = np.linalg.norm(rays, axis=-1, keepdims=True)
        rays /= norm
        return rays

class DepthSimulator:
    def __init__(self):
        # RGB Sensor: 800x600, 95x75 deg
        self.rgb_sensor = SensorConfig(800, 600, 95, 75)
        
        # Depth Sensor: 224x100, 100x15 deg
        # "Rigged in same plate... calibrated extrinsic matrix"
        # We'll assume a small baseline offset if not specified, 
        # but User said "same plate", often implies very close. 
        # Let's assume Identity extrinsic for now unless specified.
        self.depth_sensor = SensorConfig(224, 100, 100, 15)
        
    def generate_sparse_pattern_mask(self, angular_spacing_deg=1.5):
        """
        Generates a binary mask for the Depth Sensor (224x100) 
        where pixels corresponding to the 1.5 deg grid are True.
        """
        rays = self.depth_sensor.get_pixel_rays() # [H, W, 3]
        
        # Convert rays to spherical coords (azimuth, elevation)
        # z is forward (axis 2), x is right (axis 0), y is down (axis 1)
        # theta (azimuth) = arctan2(x, z)
        # phi (elevation) = arcsin(y) -- assuming y is normalized
        
        x = rays[..., 0]
        y = rays[..., 1]
        z = rays[..., 2]
        
        azimuth = np.arctan2(x, z) # rad
        elevation = np.arcsin(y)   # rad (approx for small angles)
        
        az_deg = np.rad2deg(azimuth)
        el_deg = np.rad2deg(elevation)
        
        # Define grid centers
        # We need to bin these rays into buckets of size 'angular_spacing_deg'
        # Or simpler: Check if the ray is "close enough" to a grid line?
        # User said: "uniformly distributed sparse dot pattern which has spacing 1.5deg"
        # This usually means a diffractive optical element (DOE).
        # We can simulate this by sampling specific pixels that align with this grid.
        
        # Let's just create a modulo mask on the angles.
        # But wait, pixels are discrete. We should pick the pixel *closest* to each grid point.
        
        mask = np.zeros((self.depth_sensor.height, self.depth_sensor.width), dtype=bool)
        
        # Range of FOV
        h_min, h_max = -self.depth_sensor.hfov/2, self.depth_sensor.hfov/2 # rad
        v_min, v_max = -self.depth_sensor.vfov/2, self.depth_sensor.vfov/2 # rad
        
        spacing = np.deg2rad(angular_spacing_deg)
        
        az_grid = np.arange(h_min, h_max, spacing)
        el_grid = np.arange(v_min, v_max, spacing)
        
        # For each grid point, project to image plane and set pixel
        # u = fx * x/z + cx
        # x/z = tan(azimuth)
        # v = fy * y/z + cy
        # y/z = tan(elevation) / cos(azimuth) ... approximate
        
        # Rigorous projection:
        # P = [sin(az)cos(el), sin(el), cos(az)cos(el)] (approx)
        # Actually easier:
        # x = tan(az) * z
        # y = tan(el) * z  (if we treat azimuth/elevation as separate separable scanners like a raster, usually DOE is spherical though)
        
        K = self.depth_sensor.K
        count = 0
        for az in az_grid:
            for el in el_grid:
                # Direction vector
                # This depends on spherical definition. 
                # Let's assume standard spherical:
                # x = sin(az) * cos(el)
                # y = sin(el)
                # z = cos(az) * cos(el)
                
                d = np.array([np.sin(az)*np.cos(el), np.sin(el), np.cos(az)*np.cos(el)])
                
                # Project
                # pixel = K @ d, then normalize by z (which is d[2])
                uv_h = K @ d
                u = int(uv_h[0] / uv_h[2])
                v = int(uv_h[1] / uv_h[2])
                
                if 0 <= u < self.depth_sensor.width and 0 <= v < self.depth_sensor.height:
                    mask[v, u] = True
                    count += 1
        
        print(f"Generated sparse mask with {count} valid points (Target ~600)")
        return mask

    def simulate_from_dense_depth(self, dense_depth_800x600):
        """
        Takes a dense depth map (800x600) matching RGB extrinsics.
        Returns: 224x100 sparse depth map with 1.5 deg pattern.
        """
        # 1. Project 800x600 pixels to 3D points
        # For simplicity, assuming Identity Extrinsic between RGB and depth for now
        # (User said "same plate", likely small offset, but we map 1:1 for this training task)
        
        # We need to sample the dense_depth at the locations corresponding to the Sparse Mask.
        # Sparse Mask is 224x100.
        # We generated 'mask' which tells us WHICH pixels in 224x100 are valid.
        # But we need the DEPTH value there.
        # If Extrinsics are Identity, FOV is different.
        # Depth Sensor: 100x15 deg
        # RGB Sensor: 95x75 deg
        
        # The Depth Sensor is WIDER (100 deg) than RGB (95 deg) horizontally?
        # That means some depth points are outside RGB frame.
        # We can ignoring them or pad.
        
        # Backward Warp:
        # For each pixel (u_d, v_d) in Depth Sensor:
        #   Ray D = K_d^-1 * [u_d, v_d, 1]
        #   Project to RGB Image Plane:
        #   [u_rgb, v_rgb] = K_rgb * D (assuming R=I, t=0)
        
        # Precompute this mapping grid?
        if not hasattr(self, 'map_x'):
            H_d, W_d = self.depth_sensor.height, self.depth_sensor.width
            rays_d = self.depth_sensor.get_pixel_rays() # (H, W, 3) normalized
            
            # Project rays to RGB image plane
            # P_rgb = K_rgb * rays_d
            # But rays_d is unit vector. 
            # We assume depth is at distance Z along that ray.
            # Actually, if frame is same, direction vector is same.
            
            # K_rgb projection
            # uv_rgb = K_rgb @ rays_d
            # But we need to handle shape (H, W, 3) 
            # K is (3,3)
            
            K_rgb = self.rgb_sensor.K
            # Reshape rays to (N, 3)
            rays_flat = rays_d.reshape(-1, 3).T
            uv_hom = K_rgb @ rays_flat # (3, N)
            
            u_rgb = uv_hom[0] / uv_hom[2]
            v_rgb = uv_hom[1] / uv_hom[2]
            
            self.map_x = u_rgb.reshape(H_d, W_d).astype(np.float32)
            self.map_y = v_rgb.reshape(H_d, W_d).astype(np.float32)
            
        # Remap
        import cv2
        # Use Nearest to avoid interpolating depth across edges, or Linear?
        # Linear is okay for dense.
        depth_in_depth_sensor_view = cv2.remap(dense_depth_800x600, self.map_x, self.map_y, interpolation=cv2.INTER_LINEAR)
        
        # Apply mask
        mask = self.generate_sparse_pattern_mask(1.5)
        sparse_depth = depth_in_depth_sensor_view * mask
        
        return sparse_depth

if __name__ == "__main__":
    # Test
    sim = DepthSimulator()
    mask = sim.generate_sparse_pattern_mask(1.5)
    
    # Visualization (ASCII) or save
    import matplotlib.pyplot as plt
    plt.imshow(mask, interpolation='nearest')
    plt.title(f"Sparse Mask (224x100) - {np.sum(mask)} points")
    plt.savefig("sparse_mask_debug.png")
