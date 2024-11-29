import os
from pathlib import Path
import cv2
import imageio
import numpy as np
import torch
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
import imageio.v3 as iiov3
import matplotlib.pyplot as plt


def load_image(imfile, device="cpu", resize_h=None, resize_w=None):
    img_pil = Image.open(imfile)
    if resize_h is not None:
        img_pil = img_pil.resize((resize_w, resize_h), Image.LANCZOS)
    img = np.array(img_pil).astype(np.uint8)
    if len(img.shape) == 2:
        img = torch.from_numpy(img).float()
    else:
        img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)


class InputPadder:
    """Pads images such that dimensions are divisible by 8"""

    def __init__(self, dims, mode="sintel"):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == "sintel":
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]
    
    
def resize_tensor_frames_lanczos(frames, h, w):
    resized_frames = []
    for frame in frames:
        resized_frame_pil = transforms.ToPILImage()(frame).resize((w, h), resample=Image.LANCZOS)
        resized_frames.append(transforms.ToTensor()(resized_frame_pil))
    return torch.stack(resized_frames).to(frames.device)


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device), indexing='ij')
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)



def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True, mode=mode)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def load_video(video_folder: str, resize=None, num_frames=None, to_tensor=True):
    """
    Loads video from folder, resizes frames as desired, and outputs video tensor.

    Args:
        video_folder (str): folder containing all frames of video, ordered by frames index.
        resize (tuple): desired H' x W' dimensions. Defaults to (432, 768).
        num_frames (int): number of frames in video. Defaults to None.

    Returns:
        return_dict: Dictionary video tensor of shape: T x 3 x H' x W'.
    """
    path = Path(video_folder)
    input_files = sorted(list(path.glob("*.jpg")) + list(path.glob("*.png")))
    input_files = input_files[:num_frames] if num_frames is not None else input_files

    resh, resw = resize if resize is not None else (None, None)
    video = []
    
    for file in input_files:
        if resize is not None:
            img = Image.open(str(file)).resize((resw, resh), Image.LANCZOS)
            img = transforms.ToTensor()(img) if to_tensor else img
            video.append(img)
        else:
            img = Image.open(str(file))
            img = transforms.ToTensor()(img) if to_tensor else img
            video.append(img)
    
    return torch.stack(video) if to_tensor else video


def save_video(video, output_path, fps=30):
    """
    Saves a video tensor as an mp4 video.
    
    Args:
        video (np.ndarray): video tensor of shape (T, H, W, 3).
        output_path (str): path to save video to.
        fps (int): frames per second. Defaults to 30.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    iiov3.imwrite(output_path, video, fps=fps, macro_block_size=None)


def save_video_frames(video, folder: str, matplotlib=False):
    """
    Saves a video as a folder of frames.
    
    Args:
        video (np.ndarray): video tensor of shape (T, H, W, 3).
        folder (str): folder to save video frames to.
        matplotlib (bool): whether to use matplotlib to save frames. Defaults to False.
        
    Returns:
        path (str): path to folder containing video frames.
    """
    path = Path(folder)
    path.mkdir(exist_ok=True, parents=True)
    if matplotlib:
        assert video.shape[-1] == 1, "Only grayscale videos with colormap from matplotlib are allowed"
        fig = plt.figure()
        fig.set_size_inches(26, 15, forward=True)
        ax = plt.subplot(111)
        plt.axis("off")
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.axis("off")

        im = ax.imshow(video[0], cmap="jet")
        plt.savefig(path / f"{0:05d}.jpg")
        for idx, frame in enumerate(video):
            im.set_data(frame)
            plt.savefig(path / f"{idx:05d}.jpg")
    else:
        for idx, frame in enumerate(video):
            imageio.imwrite(path / f"{idx:05d}.jpg", frame)

    return path


def resize_flow(flow, newh, neww):
    oldh, oldw = flow.shape[2:]
    flow_resized = cv2.resize(flow[0].permute(1, 2, 0).cpu().detach().numpy(), (neww, newh), interpolation=cv2.INTER_LINEAR)
    flow_resized[:, :, 0] *= neww / oldw
    flow_resized[:, :, 1] *= newh / oldh
    flow_resized = torch.from_numpy(flow_resized).permute(2, 0, 1).to(flow.device).unsqueeze(0)
    return flow_resized


def get_points_on_an_interval_grid(interval, interp_shape, device="cpu"):
    grid_y = torch.arange(0, interp_shape[0], interval, device=device) # Y
    grid_x = torch.arange(0, interp_shape[1], interval, device=device) # X
    grid_y, grid_x = torch.meshgrid(grid_y, grid_x)
    xy = torch.stack([grid_x, grid_y], dim=-1).to(device) # Y x X x 2, where 2 is the (x, y) coordinates
    xy = xy.reshape(-1, 2).unsqueeze(0) # 1 x (Y * X) x 2
    return xy


def get_grid_query_points(res_h_w, segm_mask=None, device="cuda", interval=None, query_frame=0):
    """
    res_h_w: tuple of (height, width) of the video. segm_mask: H x W tensor. interval: pixel interval between the grid points.
    """
    grid_pts = get_points_on_an_interval_grid(interval, res_h_w, device=device) # 1 x (Y * X) x 2
    
    if segm_mask is not None:
        segm_mask = F.interpolate(
            segm_mask[None, None, ...], res_h_w, mode="nearest"
        ).squeeze(0).squeeze(0) # height x width
        point_mask = segm_mask[
            (grid_pts[0, :, 1]).round().long().cpu(),
            (grid_pts[0, :, 0]).round().long().cpu(),
        ].bool()
        grid_pts = grid_pts[:, point_mask]

    query_points = torch.cat(
        [grid_pts, torch.ones_like(grid_pts[:, :, :1]) * query_frame],
        dim=2,
    )[0]

    return query_points
