import os 
import torch
import numpy as np
from tqdm import tqdm
import argparse
from dino_tracker import DINOTracker
from data.tapvid import get_query_points_from_benchmark_config
from models.model_inference import ModelInference

device = "cuda:0" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def run(args):
    dino_tracker = DINOTracker(args)
    model = dino_tracker.get_model()
    if args.iter is not None:
        model.load_weights(args.iter)

    trajectories_dir = dino_tracker.trajectories_dir
    occlusions_dir = dino_tracker.occlusions_dir
    os.makedirs(trajectories_dir, exist_ok=True)
    os.makedirs(occlusions_dir, exist_ok=True)

    
    model_inference = ModelInference(
        model=model,
        range_normalizer=dino_tracker.range_normalizer,
        anchor_cosine_similarity_threshold=dino_tracker.config['anchor_cosine_similarity_threshold'],
        cosine_similarity_threshold=dino_tracker.config['cosine_similarity_threshold'],
    )

    query_points = get_query_points_from_benchmark_config(args.benchmark_pickle_path,
                                                        args.video_id,
                                                        rescale_sizes=[model.video.shape[-1], model.video.shape[-2]]) # x, y
    for frame_idx in tqdm(sorted(query_points.keys()), desc="Saving model predictions"):
        qpts_st_frame = torch.tensor(query_points[frame_idx], dtype=torch.float32, device=device) # N x 3, (x, y, t)
        trajectories_at_st_frame, occlusion_at_st_frame = model_inference.infer(query_points=qpts_st_frame, batch_size=args.batch_size) # N x T x 3, N x T
        
        np.save(os.path.join(trajectories_dir, f"trajectories_{frame_idx}.npy"), trajectories_at_st_frame[..., :2].cpu().detach().numpy())
        np.save(os.path.join(occlusions_dir, f"occlusion_preds_{frame_idx}.npy"), occlusion_at_st_frame.cpu().detach().numpy())



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/train.yaml", type=str)
    parser.add_argument("--data-path", default="./dataset/libby", type=str)
    parser.add_argument("--benchmark-pickle-path", default="./dataset/libby/benchmark.pkl", type=str)
    parser.add_argument("--video-id", type=int, default=0)
    parser.add_argument("--iter", type=int, default=None, help="Iteration number of the model to load, if None, the last checkpoint is loaded.")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    run(args)
