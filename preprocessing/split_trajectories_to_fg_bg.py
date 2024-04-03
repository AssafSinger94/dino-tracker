from pathlib import Path
import torch
from torch.nn.functional import interpolate
import numpy as np
from PIL import Image
import argparse
device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_start_end(trajectories):
    """Generates a start and end point for each trajectory

    Args:
        trajectories (torch.Tensor): A tensor of size Nxtx2 that contains point trajectory where optical flow was consistent and "nan" where it wasn't
    """
    mask = trajectories.isnan().any(dim=-1)

    # Generate start points
    mask_shifted_right = mask.roll(1, dims=1)
    mask_shifted_right[:, 0] = True

    first_timestep_mask = ~mask & mask_shifted_right
    first_timestep = first_timestep_mask.nonzero()[:, 1]
    first_timestep_indices = first_timestep_mask.nonzero()[:, 0]

    # Generate end points
    mask_shifted_left = mask.roll(-1, dims=1)
    mask_shifted_left[:, -1] = True

    last_timestep_mask = ~mask & mask_shifted_left
    last_timestep = last_timestep_mask.nonzero()[:, 1]

    # Combine
    start_end = torch.stack([first_timestep_indices, first_timestep, last_timestep], dim=1)

    return start_end, first_timestep_mask


def load_masks(masks_path, h_resize=476, w_resize=854):
    masks = []
    input_files = sorted(list(Path(masks_path).glob("*.jpg")) + list(Path(masks_path).glob("*.png")))
    for mask_path in input_files:
        mask = np.array(Image.open(mask_path).convert("L")) # convert to grayscale if necessary
        masks.append(mask)
    masks = np.stack(masks) # B x H x W
    h_resize = masks.shape[1] if h_resize is None else h_resize
    w_resize = masks.shape[2] if w_resize is None else w_resize
    masks = torch.from_numpy(masks).unsqueeze(1) # B x 1 x H x W
    masks = interpolate(masks,
                        size=(h_resize, w_resize),
                        mode="nearest")
    masks = masks[:, 0, :, :].numpy() # B x h_resize x w_resize
    return masks


def mask_filter_trajectories(traj_path, masks_path, out_path, filter_bg=False):
    trajectories = torch.load(traj_path, map_location="cpu")
    masks = load_masks(masks_path)
    
    batch_size = 1_000_000
    is_valid_traj_list = []
    masks_t = torch.from_numpy(masks).cuda()
    for i in range(0, trajectories.shape[0], batch_size):
        end = min(i + batch_size, trajectories.shape[0])
        trajectories_batch = trajectories[i:end].to(device).clone()
        
        trajectories_start_end, first_timestep_mask = generate_start_end(trajectories_batch)
        start_indices = first_timestep_mask.int().argmax(dim=1)
        traj_start_points = trajectories_batch[torch.arange(trajectories_batch.shape[0]), start_indices].round().int()
        # print(masks_t.shape, traj_start_points.shape)
        masks_at_start = masks_t[start_indices, traj_start_points[:, 1], traj_start_points[:, 0]]
        is_valid_traj = masks_at_start == 0 if filter_bg else masks_at_start > 0
        is_valid_traj_list.append(is_valid_traj)
        del trajectories_batch, trajectories_start_end, first_timestep_mask, start_indices, traj_start_points, masks_at_start
        
    is_valid_traj = torch.cat(is_valid_traj_list, dim=0).cpu()
    filtered_trajs = trajectories[is_valid_traj]
    torch.save(filtered_trajs, out_path)
    print(f"Saved {out_path}, shape: {filtered_trajs.shape}")

def split_trajectories_to_fg_bg(args):
    mask_filter_trajectories(args.traj_path, args.fg_masks_path, args.fg_traj_path, filter_bg=False)
    mask_filter_trajectories(args.traj_path, args.fg_masks_path, args.bg_traj_path, filter_bg=True)
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_path", default="")
    parser.add_argument("--fg_masks_path", default="")
    parser.add_argument("--fg_traj_path", default="")
    parser.add_argument("--bg_traj_path", default="")
    args = parser.parse_args()
    split_trajectories_to_fg_bg(args)
