import torch
import torch.nn as nn
from data.dataset import RangeNormalizer
from einops import rearrange
import math
from models.networks.conv_norm import NormalizedConv2d


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# copied from OmniMotion
def gen_grid(h_start, w_start, h_end, w_end, step_h, step_w, device, normalize=False, homogeneous=False):
    """Generate a grid of coordinates in the image frame.
    Args:
        h, w: height and width of the grid.
        device: device to put the grid on.
        normalize: whether to normalize the grid coordinates to [-1, 1].
        homogeneous: whether to return the homogeneous coordinates. homogeneous coordinates are 3D coordinates.
    Returns:"""
    if normalize:
        lin_y = torch.linspace(-1., 1., steps=h_end, device=device)
        lin_x = torch.linspace(-1., 1., steps=w_end, device=device)
    else:
        lin_y = torch.arange(h_start, h_end, step=step_h, device=device)
        lin_x = torch.arange(w_start, w_end, step=step_w, device=device)
    grid_y, grid_x = torch.meshgrid((lin_y, lin_x))
    grid = torch.stack((grid_x, grid_y), -1)
    if homogeneous:
        grid = torch.cat([grid, torch.ones_like(grid[..., :1])], dim=-1)
    return grid  # [h, w, 2 or 3]


class TrackerHead(nn.Module):
    def __init__(self,
                 use_cnn_refiner=True,
                 
                 in_channels=1,
                 hidden_channels=16,
                 out_channels=1,
                 kernel_size=3,
                 stride=1,
                 
                 patch_size=14,
                 step_h=14,
                 step_w=14,
                 argmax_radius=35,
                 video_h=480,
                 video_w=640):
        super(TrackerHead, self).__init__()
        
        self.use_cnn_refiner = use_cnn_refiner
        padding = kernel_size // 2
        self.cnn_refiner = nn.Sequential(
            NormalizedConv2d(in_channels, hidden_channels, kernel_size, stride, padding=padding),
            nn.ReLU(inplace=True),
            NormalizedConv2d(hidden_channels, out_channels, kernel_size, stride, padding=padding),
        ) if self.use_cnn_refiner else nn.Identity()
        
        self.softmax = nn.Softmax(dim=2)
        self.argmax_radius = argmax_radius
        self.patch_size = patch_size
        self.step_h = step_h
        self.step_w = step_w
        self.video_h=video_h
        self.video_w=video_w
    
    def soft_argmax(self, heatmap, argmax_indices):
        """
        heatmap: shape (B, H, W)
        """
        h_start = self.patch_size // 2
        w_start = self.patch_size // 2
        h_end = ((self.video_h - 2 * h_start) // self.step_h) * self.step_h + h_start + math.ceil(self.step_h / 2)
        w_end = ((self.video_w - 2 * w_start) // self.step_w) * self.step_w + w_start + math.ceil(self.step_w / 2)
        grid = gen_grid(h_start=h_start, w_start=w_start, h_end=h_end, w_end=w_end, step_h=self.step_h, step_w=self.step_w,
                        device=heatmap.device, normalize=False, homogeneous=False) # shape (H, W, 2)
        grid = grid.unsqueeze(0).repeat(heatmap.shape[0], 1, 1, 1) # stack and repeat grid to match heatmap shape (B, H, W, 2)
        
        row, col = argmax_indices
        argmax_coord = torch.stack((col*self.step_w+w_start, row*self.step_h+h_start), dim=-1) # (x,y) coordinates, shape (B, 2)
        
        # generate a mask of a circle of radius radius around the argmax_coord (B, 2) in heatmap (B, H, W, 2)
        mask = torch.norm((grid - argmax_coord.unsqueeze(1).unsqueeze(2)).to(torch.float32), dim=-1) <= self.argmax_radius # shape (B, H, W)
        heatmap = heatmap * mask
        hm_sum = torch.sum(heatmap, dim=(1, 2)) # B
        hm_zero_indices = hm_sum < 1e-8
        
        # for numerical stability
        if sum(hm_zero_indices) > 0:
            uniform_w = 1 / mask[hm_zero_indices].sum(dim=(1,2))
            heatmap[hm_zero_indices] += uniform_w[:, None, None]
            heatmap[hm_zero_indices] = heatmap[hm_zero_indices] * mask[hm_zero_indices]
            hm_sum[hm_zero_indices] = torch.sum(heatmap[hm_zero_indices], dim=(1, 2))

        point = torch.sum(grid * heatmap.unsqueeze(-1), dim=(1, 2)) / hm_sum.unsqueeze(-1) # shape (B, 2)

        return point
    
    def softmax_heatmap(self, hm):
        b, c, h, w = hm.shape
        hm_sm = rearrange(hm, "b c h w -> b c (h w)") # shape (B, 1, H*W)
        hm_sm = self.softmax(hm_sm) # shape (B, 1, H*W)
        hm_sm = rearrange(hm_sm, "b c (h w) -> b c h w", h=h, w=w) # shape (B, 1, H, W)
        return hm_sm
    
    def forward(self, cost_volume):
        """
        cost_volume: shape (B, C, H, W)
        """
        
        range_normalizer = RangeNormalizer(shapes=(self.video_w, self.video_h)) # shapes are (W, H), correpsonding to (x, y) coordinates
        
        # crop heatmap around argmax point
        argmax_flat = torch.argmax(rearrange(cost_volume[:, 0], "b h w -> b (h w)"), dim=1)
        argmax_indices = (argmax_flat // cost_volume[:, 0].shape[-1], argmax_flat % cost_volume[:, 0].shape[-1])

        refined_heatmap = self.softmax_heatmap(self.cnn_refiner(cost_volume)) # shape (B, 1, H, W)
        point = self.soft_argmax(refined_heatmap.squeeze(1),
                                 argmax_indices) # shape (B, 2), (x,y) coordinates
        return range_normalizer(point, dst=(-1,1), dims=[0, 1]) # shape (B, 2)
