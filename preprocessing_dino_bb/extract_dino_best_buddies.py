import os 
import torch
import torchvision.transforms as T
import argparse
from tqdm import tqdm
from einops import rearrange
from preprocessing_dino_bb.dino_bb_utils import create_meshgrid


device = "cuda:0" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def run(args):
    dino_embed_video_path = args.dino_emb_path
    h, w = args.h, args.w
    out_path = args.out_path
    
    best_buddies = {}
    coords_grid = create_meshgrid(h, w, step=args.stride).to(device)
    
    features = torch.load(dino_embed_video_path) # T x C x H x W
    features = rearrange(features, 't c h w -> t (h w) c')
    torch.cuda.empty_cache()

    t = features.shape[0]
    for source_t in tqdm(range(t), desc="source time"):
        for target_t in tqdm(range(t), desc="target time"):
            if source_t == target_t:
                continue
            
            source_features = features[source_t].to(device) # (h x w) x c
            target_features = features[target_t].to(device) # (h x w) x c
            feature_range = torch.arange(source_features.shape[0]).to(device)

            affinity = torch.einsum("nc,mc->nm", source_features, target_features) # n x m
            affinity = affinity / torch.clamp(source_features.norm(dim=1)[:, None] * target_features.norm(dim=1)[None, ...], min=1e-08) # n x m
            affinity_source_max = torch.argmax(affinity, dim=1) # n
            affinity_target_max = torch.argmax(affinity, dim=0) # m
            source_bb_indices = feature_range == affinity_target_max[affinity_source_max]
            target_bb_indices = affinity_source_max[source_bb_indices]

            source_coords = coords_grid[source_bb_indices]
            target_coords = coords_grid[target_bb_indices]
            affinities = affinity[feature_range[source_bb_indices], target_bb_indices]
            
            best_buddies[f'{source_t}_{target_t}'] = {
                "source_coords": source_coords,
                "target_coords": target_coords,
                "cos_sims": affinities
            }
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(best_buddies, out_path)
    print(f"Saved best buddies to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino-emb-path", type=str, required=True)
    parser.add_argument("--h", type=int, required=True)
    parser.add_argument("--w", type=int, required=True)
    parser.add_argument("--stride", type=int, default=7)
    parser.add_argument("--out-path", type=str, required=True)
    args = parser.parse_args()
    run(args)
