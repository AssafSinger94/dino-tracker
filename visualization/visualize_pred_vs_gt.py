import argparse
import os
import pickle
from tqdm import tqdm
import cv2
import numpy as np

from data.data_utils import load_video, save_video
from data.tapvid import get_video_config_by_video_id
from utils import add_config_paths
from visualization.viz_utils import get_colors

def overlay_cross_cv2(image, point, color, thickness, r=6):
    # Draw a cross at the specified coordinate
    x, y = point
    cv2.line(image, (x-r, y-r), (x+r, y+r), color=color, thickness=thickness)
    cv2.line(image, (x-r, y+r), (x+r, y-r), color=color, thickness=thickness)

    return image

def overlay_pred_gt_on_frame(img, color, pred_point, gt_point, pred_occluded=0, gt_occluded=0, thickness=4, radius=8, cross_size=8):
    pred_point = tuple(pred_point.astype(int))
    gt_point = tuple(gt_point.astype(int))
    pred_occluded = bool(pred_occluded)
    gt_occluded = bool(gt_occluded)

    red_color = (255, 0, 0) # color for displacement lines

    if (not pred_occluded) and (not gt_occluded): 
        cv2.line(img, pred_point, gt_point, color=red_color, thickness=thickness)
        cv2.circle(img, center=pred_point, radius=radius, color=color, thickness=-1)
    elif (not pred_occluded) and (gt_occluded):
        overlay_cross_cv2(img, pred_point, color=color, thickness=thickness, r=cross_size)
    elif (pred_occluded) and (not gt_occluded):
        cv2.line(img, pred_point, gt_point, color=red_color, thickness=thickness//2)
        cv2.circle(img, center=pred_point, radius=radius, color=color, thickness=2)
    
    return img

def visualize_trajectories_with_gt(video, pred_trajectories, gt_trajectories, pred_occluded=None, gt_occluded=None, 
                                   thickness=4, radius=8, cross_size=8, badja_vis_type=False):

    assert pred_trajectories.shape == gt_trajectories.shape, \
        f"pred and gt trajectories must be the same shape, pred.shape={pred_trajectories.shape}, gt.shape={gt_trajectories.shape}"

    colormap = get_colors(num_colors=len(pred_trajectories), seed=0, without_red=True)
    frames_for_vis = range(len(video))
    
    if badja_vis_type:
        # filter only frames where less then 60% of gt_points are occluded
        frames_for_vis = [i for i in range(len(video)) if (((gt_trajectories[:, i, :] < 1).all(axis=-1)).mean() < 0.6)]
    
    frames = []
    for cur_frame in tqdm(frames_for_vis):
        img = np.copy(video[cur_frame])
        for idx, (pred_trajectory, gt_trajectory, pred_occluded_trajectory, gt_occluded_trajectory) in enumerate(zip(pred_trajectories, gt_trajectories, pred_occluded, gt_occluded)):
            img = overlay_pred_gt_on_frame(img, colormap[idx],
                                        pred_point=pred_trajectory[cur_frame],
                                        gt_point=gt_trajectory[cur_frame],
                                        pred_occluded=pred_occluded_trajectory[cur_frame],
                                        gt_occluded=gt_occluded_trajectory[cur_frame],
                                        thickness=thickness,
                                        radius=radius,
                                        cross_size=cross_size)
        frames.append(img)

    return np.stack(frames, axis=0)


def save_prediction_vs_gt(args):
    config_paths = add_config_paths(args.data_path, {})
    video_folder = config_paths["video_folder"]
    trajectories_dir = config_paths["trajectories_dir"]
    occlusions_dir = config_paths["occlusions_dir"]
    model_vis_dir = config_paths['model_vis_dir']

    # load GT data
    benchmark_data = pickle.load(open(args.benchmark_pickle_path, "rb"))
    benchmark_video_data  = get_video_config_by_video_id(benchmark_data, args.video_id)
    orig_h, orig_w = benchmark_video_data['h'], benchmark_video_data['w']
    video = load_video(video_folder)
    video = (video * 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8).copy() # T x H x W x 3, numpy.uint8, [0, 255]
    pred_h, pred_w = args.infer_res_size
    os.makedirs(model_vis_dir, exist_ok=True)

    for idx, frame_idx in tqdm(enumerate(sorted(benchmark_video_data['target_points'].keys())), desc="visualizing pred vs gt"):
        if idx > 0 and args.only_first_frame:
            break
        gt_trajectories = np.array(benchmark_video_data['target_points'][frame_idx]) # N x T x 2
        gt_occluded = np.array(benchmark_video_data['occluded'][frame_idx]) # N x T
        
        pred_trajectories = np.load(os.path.join(trajectories_dir, f"trajectories_{frame_idx}.npy"))
        pred_trajectories = pred_trajectories * np.array([orig_w / pred_w, orig_h / pred_h], dtype=np.float32) # resize tracks to video resolution
        
        if args.use_gt_occ:
            pred_occluded = gt_occluded
        else:
            assert os.path.exists(os.path.join(occlusions_dir, f"occlusion_preds_{frame_idx}.npy")), f"occlusion_preds_{frame_idx}.npy does not exist"
            pred_occluded = np.load(os.path.join(occlusions_dir, f"occlusion_preds_{frame_idx}.npy"))
        
        pred_vs_gt_video = visualize_trajectories_with_gt(video, pred_trajectories, gt_trajectories, pred_occluded, gt_occluded, 
                                                          badja_vis_type=args.badja_vis_type)

        save_video(pred_vs_gt_video, os.path.join(model_vis_dir, f"pred_vs_gt_frame_idx_{frame_idx}_fps_{args.fps}.mp4"), fps=args.fps)
        
    print("Saved to", model_vis_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="./dataset/libby", type=str, required=True)
    parser.add_argument("--benchmark-pickle-path", type=str, required=True)
    parser.add_argument("--video-id", type=int, required=True)
    parser.add_argument("--infer-res-size", type=int, nargs=2, default=(476, 854), help="Inference resolution size, (h, w). --NOTE-- change according to values in train.yaml.")
    parser.add_argument("--badja-vis-type", action="store_true", help="visualize trajectories in Badja format. only plotting the frames with GT annotations.")
    parser.add_argument("--only-first-frame", action="store_true", help="visualize only the query point for the first frame")
    parser.add_argument("--use-gt-occ", action="store_true", help="use GT occlusion for visualization")
    parser.add_argument("--fps", type=int, default=10, help="fps of the output video. fps=10 for TAP-Vid, fps=2 for BADJA.")

    args = parser.parse_args()
    save_prediction_vs_gt(args)