import argparse
from pathlib import Path
from PIL import Image
import os
import numpy as np
import torch
from kornia import morphology as morph

from utils import add_config_paths
import visualization.viz_utils_tapir as viz_utils
from data.data_utils import load_video, save_video

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# The inlier point threshold for ransac, specified in normalized coordinates
# (points are rescaled to the range [0, 1] for optimization).
ransac_inlier_threshold = 0.07  # @param {type: "number"}
# What fraction of points need to be inliers for RANSAC to consider a trajectory
# to be trustworthy for estimating the homography.
ransac_track_inlier_frac = 0.95  # @param {type: "number"}
# After initial RANSAC, how many refinement passes to adjust the homographies
# based on tracks that have been deemed trustworthy.
num_refinement_passes = 2  # @param {type: "number"}
# # After homographies are estimated, consider points to be outliers if they are
# # further than this threshold.
# foreground_inlier_threshold = 0.07  # @param {type: "number"}
# # After homographies are estimated, consider tracks to be part of the foreground
# # if less than this fraction of its points are inliers.
# foreground_frac = 0.6  # @param {type: "number"}


def filter_bg_trajectories_for_homographies(bg_tracjectories, bg_trajectories_count=500, canonical_frame=None, min_len=10):
    N, T, _ = bg_tracjectories.shape
    if canonical_frame is None:
        canonical_frame = bg_tracjectories.shape[1] // 2
    # for each frame, find the longest trajectories that are not nan in that frame and the canonical frame
    of_len = (~(bg_tracjectories.isnan().any(dim=-1))).sum(dim=-1)
    of_idx_list = []
    bg_tracjectories_count_per_frame = bg_trajectories_count // T # have some redundancy of trajectories in case of repeating trajectories
    for frame_idx in range(T):
        # find all traejctories that are not nan in that frame and the canonical frame
        valid_of_idx = (~(bg_tracjectories[:, frame_idx].isnan().any(dim=-1)) & ~(bg_tracjectories[:, canonical_frame].isnan().any(dim=-1)))
        # search for valid trajectories that are longer than 10 frames
        long_valid_of_idx = torch.where((of_len * valid_of_idx.float()) > min_len)[0]
        if len(long_valid_of_idx) < bg_tracjectories_count_per_frame:
            print(f"frame {frame_idx} and canonical frame {canonical_frame} have less than {bg_tracjectories_count_per_frame} valid trajectories for homography estimation.")
            long_valid_of_idx = torch.where((of_len * valid_of_idx.float()) > 5)[0]
        # randomly sample bg_tracjectories_count_per_frame trajectories
        long_valid_of_idx = long_valid_of_idx[torch.randperm(len(long_valid_of_idx))[:bg_tracjectories_count_per_frame]]
        of_idx_list.append(long_valid_of_idx)
    of_idx_list = torch.stack(of_idx_list, dim=1)
    # remove duplicate trajectories
    of_idx_list = torch.unique(of_idx_list.reshape(-1))
    return bg_tracjectories[of_idx_list]

@torch.no_grad()
def run(args):
    config_paths = add_config_paths(args.data_path, {})
    video_folder = config_paths["video_folder"]
    masks_path = Path(config_paths["masks_path"])
    segm_mask_path = sorted(list(Path(masks_path).glob("*.jpg")) + list(Path(masks_path).glob("*.png")))[args.vis_start_frame]
    bg_of_trajectories_path = config_paths["bg_trajectories_file"]
    trajs_path = os.path.join(config_paths["grid_trajectories_dir"], f"grid_trajectories.npy")
    occ_path = os.path.join(config_paths["grid_occlusions_dir"], f"grid_occlusions.npy")
    model_vis_dir = config_paths['model_vis_dir']
    

    video = load_video(video_folder, num_frames=300) # T x 3 x H x W, torch.float32, [0, 1]
    video = (video.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8) # T x H x W x 3, numpy.uint8, [0, 255]
    video_h, video_w = video.shape[1], video.shape[2]

    tracks = np.load(trajs_path) # [N, T, 2]
    if args.infer_res_size is not None:
        pred_h, pred_w = args.infer_res_size
        tracks = tracks * np.array([video_w / pred_w, video_h / pred_h], dtype=np.float32) # resize tracks to video resolution
    
    try:
        occluded = np.load(occ_path).astype(np.int32) # [N, T]
    except:
        print(f"{occ_path} does not exist, marking all points as visible ---")
        occluded = np.zeros(tracks.shape[:-1])
    
    # load segmentation mask
    segm_mask = torch.from_numpy(np.array(Image.open(segm_mask_path).convert("L"))).bool().float().cpu().numpy()
    if segm_mask.shape != (video_h, video_w):
        segm_mask = torch.nn.functional.interpolate(torch.from_numpy(segm_mask).unsqueeze(0).unsqueeze(0), size=(video_h, video_w), mode="nearest").squeeze().cpu().numpy() # H x W
    
    if args.erosion_kernel_size is not None:
        erosion_kernel = torch.ones(args.erosion_kernel_size, args.erosion_kernel_size)
        segm_mask = morph.erosion(torch.from_numpy(segm_mask).float()[None, None], erosion_kernel).bool().squeeze()  # Erosion, H x W

    coords = tracks[:,0].round().astype(np.int32)
    is_fg = (segm_mask[coords[:, 1], coords[:, 0]] > 0)

    vis_start_frame = args.vis_start_frame # default 0
    vis_end_frame = args.vis_end_frame if args.vis_end_frame is not None else video.shape[0]
    
    video = video[vis_start_frame: vis_end_frame]
    tracks = tracks[:, vis_start_frame:vis_end_frame]
    occluded = occluded[:, vis_start_frame:vis_end_frame]

    
    os.makedirs(model_vis_dir, exist_ok=True)
    dotted_vis_name = f"dotted_tracks_fps_{args.fps}.mp4" if args.erosion_kernel_size is None else f"dotted_tracks_erosion_kernel_{args.erosion_kernel_size}_fps_{args.fps}.mp4"
    tracks_video_no_trace = viz_utils.plot_tracks_v2(video, tracks[is_fg], occluded[is_fg], rainbow_colors=True, point_size=args.point_size)
    print(tracks_video_no_trace.shape, video.shape, tracks[is_fg].shape, occluded[is_fg].shape)
    save_video(tracks_video_no_trace, os.path.join(model_vis_dir, dotted_vis_name), fps=args.fps)
    
    if args.plot_trails:
        # load background optical flow trajectories, used for homography estimation
        bg_of_trajectories = torch.load(bg_of_trajectories_path).to(device)
        bg_of_trajectories = bg_of_trajectories[:, vis_start_frame:vis_end_frame]
        bg_of_tracks = filter_bg_trajectories_for_homographies(bg_of_trajectories, canonical_frame=args.canonical_frame)
        bg_of_occluded = bg_of_tracks.isnan().any(dim=-1).int().cpu().numpy()
        bg_of_tracks = np.nan_to_num(bg_of_tracks.cpu().numpy(), nan=0)

        of_h, of_w = args.of_res_size
        bg_of_tracks = bg_of_tracks * np.array([video_w / of_w, video_h / of_h], dtype=np.float32) # resize tracks to video resolution

        homogs, err, canonical = viz_utils.get_homographies_wrt_frame(
            bg_of_tracks,
            bg_of_occluded,
            [video_w, video_h],
            thresh=ransac_inlier_threshold,
            outlier_point_threshold=ransac_track_inlier_frac,
            num_refinement_passes=num_refinement_passes,
            reference_frame=args.canonical_frame
        )

        rainbow_video = viz_utils.plot_tracks_tails(
            video,
            tracks[is_fg],
            occluded[is_fg],
            homogs, 
            point_size=args.point_size,
            linewidth=args.linewidth,
            marker="D",
        )
        rainbow_vis_name = f"rainbow_erosion_kernel_{args.erosion_kernel_size}_fps_{args.fps}.mp4" if args.erosion_kernel_size is not None else f"rainbow_fps_{args.fps}.mp4"
        save_video(rainbow_video, os.path.join(model_vis_dir, rainbow_vis_name), fps=args.fps)
    
    print("Saved to", model_vis_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="./dataset/libby", type=str)
    parser.add_argument("--infer-res-size", type=int, nargs=2, default=(476, 854), help="Inference resolution size, (h, w). --NOTE-- change according to values in train.yaml.")
    parser.add_argument("--of-res-size", type=int, nargs=2, default=(476, 854), help="Optical flow resolution size, (h, w). --NOTE-- change according to values in preprocess.yaml.")
    parser.add_argument("--erosion-kernel-size", type=int, default=None, help="size of the erosion kernel for the segmentation mask. If None, no erosion is applied.")
    parser.add_argument("--vis-start-frame", type=int, default=0, help="should be same as start_frame in inference_grid.py")
    parser.add_argument("--vis-end-frame", type=int, default=None)
    parser.add_argument("--canonical-frame", type=int, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--point-size", type=int, default=40)
    parser.add_argument("--linewidth", type=float, default=1.5)
    parser.add_argument("--plot-trails", action="store_true", default=False, help="Plot rainbow trails using homographies.")
    args = parser.parse_args()
    
    run(args)