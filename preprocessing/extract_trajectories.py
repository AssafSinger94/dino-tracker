import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from einops import rearrange, repeat
from tqdm import tqdm
from tqdm.contrib import tzip
from matplotlib import pyplot as plt
import os

device = "cuda:0" if torch.cuda.is_available() else "cpu"


from data.data_utils import (
    load_image,
    InputPadder,
    coords_grid,
    bilinear_sampler,
    resize_tensor_frames_lanczos,
    resize_flow
)
from utils import bilinear_interpolate_video

torch.autograd.set_grad_enabled(False)


@torch.no_grad()
def get_flows_with_masks(
    model,
    trasnforms,
    input_path: str,
    device: str = "cuda:0",
    threshold: float = 1,
    add_missing_forward_warp: bool = True,
    infer_res_size=None
):
    # reading frames
    input_folder = Path(input_path)
    images = list(input_folder.glob("*.jpg")) + list(input_folder.glob("*.png"))
    images = sorted(images)
    h, w = load_image(images[0], device).shape[2:4] if infer_res_size is None else infer_res_size
    flows = torch.zeros((2, len(images) - 1, 2, h, w), device=device)
    masks_array = torch.zeros((len(images) + 1, h, w, 1), device=device)
    err_array = torch.zeros((len(images) + 1, h, w), device=device)
    missing_forward_warp = torch.ones((len(images) + 1, h, w), device=device, dtype=torch.bool)
    video = torch.cat([load_image(imfile, device) for imfile in images], dim=0)
    upper_bound = torch.tensor([[w, h]]).to(device) - 1

    video = video / 255  # Map [0, 255] -> [0, 1]
    if infer_res_size is not None:
        video = resize_tensor_frames_lanczos(video, h=infer_res_size[0], w=infer_res_size[1])
    padder = InputPadder(video.shape)
    raft_video = padder.pad(video)[0]

    raft_video, _ = trasnforms(raft_video, raft_video)

    coords = rearrange(coords_grid(1, h, w, device=device), "1 d h w -> 1 h w d")

    for idx, (image1, image2) in enumerate(tzip(raft_video[:-1], raft_video[1:], desc="Calculating flows")):
        
        from_batch = torch.stack((image1, image2), dim=0) # shape
        to_batch = from_batch.flip(0)
        flow12, flow21 = model(from_batch, to_batch, num_flow_updates=24)[-1]

        flow12 = padder.unpad(flow12).unsqueeze(0) # 1, 2, H, W
        flow21 = padder.unpad(flow21).unsqueeze(0)
        if flow12.shape[0] != h or flow12.shape[1] != w:
            flow12 = resize_flow(flow12, newh=h, neww=w)
            flow21 = resize_flow(flow21, newh=h, neww=w)

        flows[:, idx] = torch.cat((flow12, flow21), dim=0)

        coords1 = coords + flow21.permute(0, 2, 3, 1)
        coords2 = coords1 + bilinear_sampler(flow12, coords1).permute(0, 2, 3, 1)

        err_array[idx + 1] = (coords - coords2).norm(dim=3)

        if add_missing_forward_warp:
            valid_warped_grid = coords + flow12.permute(0, 2, 3, 1)
            valid_warped_grid = valid_warped_grid.round().long().flatten(0, -2)
            valid_warped_grid = valid_warped_grid[
                ((valid_warped_grid >= 0) & (valid_warped_grid <= upper_bound)).all(dim=-1)
            ]
            missing_forward_warp[idx + 1, valid_warped_grid[:, 1], valid_warped_grid[:, 0]] = False


    masks_array = err_array.unsqueeze(-1) < threshold
    masks_array[0] = False  # Set mask of first frame to False as it's inconsistent with nothing

    masks_array = masks_array & ~missing_forward_warp.unsqueeze(-1)
    masks_array[0] = False  # Set mask of first frame to False as it's inconsistent with nothing

    return masks_array, flows


@torch.no_grad()
def compute_direct_flows_for_start_frame(
    model,
    trasnforms,
    input_path: str,
    device: str = "cuda:0",
    threshold: float = 1,
    starting_frame : int = 0,
    infer_res_size=None,
):
    # compute direct flow from starting_frame ot all following frames, along with the mask 
    input_folder = Path(input_path)
    images = list(input_folder.glob("*.jpg")) + list(input_folder.glob("*.png"))
    images = sorted(images)[starting_frame:] # starting from starting_frame, (video_len - starting_frame) images
    
    h, w = load_image(images[0], device).shape[2:4] if infer_res_size is None else infer_res_size
    video = torch.cat([load_image(imfile, device) for imfile in images], dim=0) # (video_len - starting_frame) x 3 x h x w
    upper_bound = torch.tensor([[w, h]]).to(device) - 1

    video = video / 255  # Map [0, 255] -> [0, 1]
    if infer_res_size is not None:
        video = resize_tensor_frames_lanczos(video, h=infer_res_size[0], w=infer_res_size[1])
    padder = InputPadder(video.shape)
    raft_video = padder.pad(video)[0]

    raft_video, _ = trasnforms(raft_video, raft_video)

    t = raft_video.shape[0] # video_len - starting_frame
    coords = rearrange(coords_grid(t-1, h, w, device=device), "b d h w -> b h w d") # (t-1) x h x w x 2

    # get direct flow from starting_frame to all following frames
    src_frame_batch = raft_video[0].unsqueeze(0).repeat(t-1, 1, 1, 1) # (t-1) x 3 x h x w
    dst_frames_batch = raft_video[1:] # (t-1) x 3 x h x w
    forward_flows = []
    backward_flows = []
    max_batch_size = 16
    for i in range(0, src_frame_batch.shape[0], max_batch_size):
        end = min(i + max_batch_size, src_frame_batch.shape[0])
        forward_flows.append(model(src_frame_batch[i:end], dst_frames_batch[i:end], num_flow_updates=24)[-1])
        backward_flows.append(model(dst_frames_batch[i:end], src_frame_batch[i:end], num_flow_updates=24)[-1])
    forward_flows = torch.cat(forward_flows, dim=0) # (t-1) x 2 x h x w
    backward_flows = torch.cat(backward_flows, dim=0) # (t-1) x 2 x h x w
    forward_flows = padder.unpad(forward_flows) # (t-1) x 2 x h x w
    backward_flows = padder.unpad(backward_flows) # (t-1) x 2 x h x w
    # no need to resize forward_flows & backward_flows, since it's already in infer_res_size
    coords1 = coords + forward_flows.permute(0, 2, 3, 1) # (t-1) x h x w x 2
    
    # convert (x,y) coordinates to (x,y,t) coordinates to indicate frame idx
    time_grid = torch.arange(coords.shape[0]).to(device) # (t-1)
    time_grid = time_grid.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, h, w, 1) # (t-1) x h x w x 1
    coords1_3d = torch.cat((coords1, time_grid), dim=-1) # (t-1) x h x w x 3
    rearrange(coords1, "t h w d -> (t h w) d") # (t-1) x h x w x 3 -> (t h w) x 3
    coords1_backward = bilinear_interpolate_video(rearrange(backward_flows, "t c h w -> 1 c t h w"), rearrange(coords1_3d, "t h w d -> (t h w) d"), h=h, w=w, t=coords.shape[0], normalize_h=True, normalize_w=True, normalize_t=True) # 1 x 2 x ((t-1)*h*w) x 1
    # convert from 1 x 2 x ((t-1)*h*w) x 1 to (t-1) x h x w x 2
    coords1_backward = coords1_backward.squeeze().permute(1, 0).reshape((coords.shape[0], h, w, 2)) # (t-1) x h x w x 2
    coords2 = coords1 + coords1_backward

    err = (coords - coords2).norm(dim=-1) # (t-1) x h x w
    mask = (err < threshold) # (t-1) x h x w
    valid_warped_grid = ((coords1 >= 0) & (coords1 <= upper_bound)).all(dim=-1)
    mask = (mask & valid_warped_grid)

    return forward_flows.permute(0, 2, 3, 1), mask.to(torch.float32) # (t-1) x h x w x 2, (t-1) x h x w


@torch.no_grad()
def save_trajectories(args):
    print(args)

    frames_path = args.frames_path
    output_path = args.output_path
    infer_res_size = args.infer_res_size # (h, w)
    threshold = args.threshold
    min_trajectory_length = args.min_trajectory_length
    filter_using_direct_flow = args.filter_using_direct_flow
    direct_flow_threshold = args.direct_flow_threshold
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # reading frames
    images = list(Path(frames_path).glob("*.jpg")) + list(Path(frames_path).glob("*.png"))
    images = sorted(images)
    h, w = load_image(images[0], device).shape[2:4] if infer_res_size is None else infer_res_size
    t = len(images)

    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device).eval()
    transforms = Raft_Large_Weights.DEFAULT.transforms()

    masks, flows = get_flows_with_masks(
        model,
        transforms,
        frames_path,
        device=device,
        add_missing_forward_warp=True,
        threshold=threshold,
        infer_res_size=infer_res_size,
    )

    fflow, bflow = flows[0:1], flows[1:2], # forward and backward flows, 1 x (t-1) x 2 x h x w

    upper_bound = torch.tensor([w, h], device=device) - 1
    lower_bound = torch.tensor([0, 0], device=device)
    all_filtered_trajectories = torch.full((0, t, 2), device="cpu", fill_value=float("nan"))
    look_behind = True


    for starting_frame in tqdm(range(t - (min_trajectory_length - 1)), leave=False):
        trajectories = torch.zeros((t - starting_frame, h, w, 2)).float().to(device) # all trajectories starting from starting_frame, (t) x h x w x 2
        coords = rearrange(coords_grid(1, h, w, device=device), "d l h w -> d h w l", d=1, l=2) # h x w x 2
        orig_coords = rearrange(coords_grid(1, h, w, device=device), "d l h w -> d h w l", d=1, l=2) # h x w x 2, for direct flow

        mask = ~masks[starting_frame]
        if look_behind:
            # look at all trajectories starting before starting_frame, and collect the valid coordinates in starting_frame that they pass through.
            past_passed = all_filtered_trajectories[:, starting_frame].to(device) # (M x 2)
            past_passed = past_passed[past_passed.isnan().any(dim=-1).logical_not()] # M' x 2
            past_passed = past_passed.round().long()
            past_passed = past_passed[((past_passed >= 0) & (past_passed <= upper_bound)).all(dim=-1)]

            not_passed_through = torch.ones_like(mask) # h x w
            not_passed_through[past_passed[:, 1], past_passed[:, 0]] = False # set to False all locations that have existing trajectories passing through them.
            mask |= not_passed_through

        trajectories[0] = torch.where(mask, coords.double(), float("nan")) # set trajectories[0] to nan in locations that already have existing trajectories (starting at previous frames) passing through them, t x h x w x 2
        
        if filter_using_direct_flow:
            dflows, dflow_masks = compute_direct_flows_for_start_frame(
                model=model,
                trasnforms=transforms,
                input_path=frames_path,
                device=device,
                threshold=threshold,
                starting_frame=starting_frame,
                infer_res_size=infer_res_size,
            )

        for idx in tqdm(range(t - 1 - starting_frame), leave=False):
            if filter_using_direct_flow:
                dflow, dflow_mask = dflows[idx], dflow_masks[idx]
                dflow_coords = orig_coords + dflow.unsqueeze(0) # 1 x h x w x 2

            flow12 = fflow[:, starting_frame + idx]
            flow21 = bflow[:, starting_frame + idx]

            flow12_warped = bilinear_sampler(flow12, coords)

            coords1 = coords + rearrange(flow12_warped, "d l h w -> d h w l", d=1, l=2)
            coords2 = coords1 + rearrange(bilinear_sampler(flow21, coords1), "d l h w -> d h w l", d=1, l=2)

            err = (coords - coords2).norm(dim=3)

            mask = mask & (err.unsqueeze(-1) < threshold) & (coords1 <= upper_bound) & (coords1 >= lower_bound)

            coords += rearrange(flow12_warped, "d l h w -> d h w l", d=1, l=2)
            if filter_using_direct_flow:
                # filter where direct flow is not consistent with trajectories[idx+1]
                err_dflow = (coords - dflow_coords).norm(dim=3)
                err_dflow = err_dflow * (dflow_mask > 0.2).float() # filter out locations where direct flow is not reliable
                mask = mask & (err_dflow.unsqueeze(-1) <  direct_flow_threshold)
            trajectories[idx + 1] = torch.where(mask, coords.double(), float("nan")) # set trajectories[idx+1] to nan in locations where the trajectory don't exist anymore (due to cycle-consistency error).

        padded_trajectories = F.pad(
            rearrange(trajectories, "t h w d -> h w d t"), (starting_frame, 0), mode="constant", value=float("nan")
        )
        padded_trajectories = rearrange(padded_trajectories, "h w d t -> (h w) t d")
        one_nan_least = padded_trajectories.isnan().any(dim=-1)
        set_nans = repeat(one_nan_least, "T t -> T t 2")
        padded_trajectories[set_nans] = float("nan")
        current_not_nan_traj = padded_trajectories[padded_trajectories.cpu().isnan().any(dim=-1).logical_not().sum(dim=-1).cuda() >= min_trajectory_length]
        all_filtered_trajectories = torch.cat([all_filtered_trajectories, current_not_nan_traj.cpu()], dim=0) # (N x t x 2), (M x t x 2) -> ((M+N) x t x 2)

    torch.save(all_filtered_trajectories, output_path)
    print(f"Saved {output_path}, shape: {all_filtered_trajectories.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-path", type=str, default=None, help="Path to frames folder")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--infer-res-size", type=int, nargs=2, default=None, help="Inference resolution size, (h, w)")
    parser.add_argument("--threshold", type=float, default=1, help="Threshold for cycle consistency error")
    parser.add_argument("--min-trajectory-length", type=int, default=2, help="Minimum trajectory length")
    parser.add_argument("--filter-using-direct-flow", action="store_true", default=False, help="Filter using direct flow")
    parser.add_argument("--direct-flow-threshold", type=float, default=None, help="Threshold for direct flow error")

    args = parser.parse_args()
    save_trajectories(args)