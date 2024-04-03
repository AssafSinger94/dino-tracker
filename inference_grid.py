import os
import numpy as np
import torch
import argparse
from dino_tracker import DINOTracker
from models.model_inference import ModelInference
from data.data_utils import get_grid_query_points

device = "cuda:0" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def run(args):
    dino_tracker = DINOTracker(args)
    dino_tracker.load_fg_masks()
    model = dino_tracker.get_model()
    if args.iter is not None:
        model.load_weights(args.iter)

    grid_trajectories_dir = dino_tracker.grid_trajectories_dir
    grid_occlusions_dir = dino_tracker.grid_occlusions_dir
    os.makedirs(grid_trajectories_dir, exist_ok=True)
    os.makedirs(grid_occlusions_dir, exist_ok=True)

    model_inference = ModelInference(
        model=model,
        range_normalizer=dino_tracker.range_normalizer,
        anchor_cosine_similarity_threshold=dino_tracker.config['anchor_cosine_similarity_threshold'],
        cosine_similarity_threshold=dino_tracker.config['cosine_similarity_threshold'],
    )

    orig_video_h, orig_video_w = dino_tracker.orig_video_res_h, dino_tracker.orig_video_res_w
    model_video_h, model_video_w = model.video.shape[-2], model.video.shape[-1]

    segm_mask = dino_tracker.fg_masks[args.start_frame].to(device) if args.use_segm_mask else None # H x W / None
    grid_query_points = get_grid_query_points((orig_video_h, orig_video_w), segm_mask=segm_mask, device=device, interval=args.interval, query_frame=args.start_frame)
    grid_query_points = grid_query_points * torch.tensor([model_video_w / orig_video_w, model_video_h / orig_video_h, 1.0]).to(device) # resizes query points to model resolution

    grid_trajectories, grid_occlusions = model_inference.infer(grid_query_points, batch_size=args.batch_size)
    np.save(os.path.join(grid_trajectories_dir, "grid_trajectories.npy"), grid_trajectories[..., :2].cpu().detach().numpy())
    np.save(os.path.join(grid_occlusions_dir, "grid_occlusions.npy"), grid_occlusions.cpu().detach().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/train.yaml", type=str)
    parser.add_argument("--data-path", default="./dataset/libby", type=str)
    parser.add_argument("--iter", type=int, default=None, help="Iteration number of the model to load, if None, the last checkpoint is loaded.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--use-segm-mask", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    run(args)
