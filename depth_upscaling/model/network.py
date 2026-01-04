import torch
import torch.nn as nn
import torch.nn.functional as F

class GuideBlock(nn.Module):
    """
    Fuses RGB features into Depth features.
    """
    def __init__(self, in_channels_rgb, in_channels_depth, out_channels):
        super().__init__()
        self.conv_rgb = nn.Conv2d(in_channels_rgb, in_channels_depth, kernel_size=1)
        self.conv_fuse = nn.Conv2d(in_channels_depth, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, rgb_feat, depth_feat):
        # RGB feature projection
        rgb_proj = self.conv_rgb(rgb_feat)
        # Element-wise addition or concatenation? 
        # Guided upsampling often uses modulation. simple addition is robust.
        fused = depth_feat + rgb_proj
        return self.act(self.bn(self.conv_fuse(fused)))

class SparseToDenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # RGB Encoder (Simulating a ResNet/MobileNet structure)
        # Level 1: 1/2
        self.rgb_enc1 = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU())
        # Level 2: 1/4
        self.rgb_enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU())
        # Level 3: 1/8
        self.rgb_enc3 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU())
        
        # Depth Encoder (Sparse Input)
        # Input 224x100 -> Projection to 800x600?
        # Actually, standard approach: Project sparse points to 800x600 buffer (mostly zeros)
        # And let the network fill holes.
        # User specified input is 224x100.
        # If we feed 224x100 directly, we need spatial alignment logic.
        # Simplest: Upsample input to 800x600 (NN) initially, then refine.
        self.input_proj = nn.Conv2d(1, 16, 3, 1, 1) # 800x600
        
        self.depth_enc1 = nn.Sequential(nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU())
        self.depth_enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU())
        self.depth_enc3 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU())
        
        # Guide Blocks
        self.guide1 = GuideBlock(32, 32, 32)
        self.guide2 = GuideBlock(64, 64, 64)
        self.guide3 = GuideBlock(128, 128, 128)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.up2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.up1 = nn.ConvTranspose2d(32, 16, 4, 2, 1)
        
        self.final = nn.Conv2d(16, 1, 3, 1, 1)
        
    def forward(self, rgb, sparse_depth_224):
        # Preprocessing: Upsample sparse depth to match RGB resolution
        # This is a naive 'alignment'. In reality, use intrinsics.
        sparse_up = F.interpolate(sparse_depth_224, size=(600, 800), mode='nearest')
        
        # Encoders
        r1 = self.rgb_enc1(rgb)       # 1/2
        r2 = self.rgb_enc2(r1)        # 1/4
        r3 = self.rgb_enc3(r2)        # 1/8
        
        d0 = self.input_proj(sparse_up)
        d1 = self.depth_enc1(d0)      # 1/2
        d1 = self.guide1(r1, d1)      # Fuse
        
        d2 = self.depth_enc2(d1)      # 1/4
        d2 = self.guide2(r2, d2)      # Fuse
        
        d3 = self.depth_enc3(d2)      # 1/8
        d3 = self.guide3(r3, d3)      # Fuse
        
        # Decoding
        x = self.up3(d3) + d2         # Skip
        x = self.up2(x) + d1          # Skip
        x = self.up1(x) + d0          # Skip
        
        out = self.final(x)
        return out

if __name__ == "__main__":
    # Test (requires torch)
    model = SparseToDenseNet()
    rgb = torch.randn(2, 3, 600, 800)
    depth = torch.randn(2, 1, 100, 224)
    out = model(rgb, depth)
    print("Output shape:", out.shape)
