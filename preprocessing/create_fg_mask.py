import argparse
import numpy as np
import torch
import torch.nn.functional as F
from data.data_utils import save_video_frames

device = "cuda:0" if torch.cuda.is_available() else "cpu"



def get_fg_mask_from_pca(
    feature_map: torch.Tensor,
    img_size,
    q=3,
    interpolation="nearest",
    normalize=True,
    fg_mask_threshold=0.4,
):
    """
    feature_map: (1, H, W, C) is the feature map of a single image.
    """
    if len(feature_map.shape) == 3:
        # make it (1, h, w, C)
        feature_map = feature_map[None]
    if normalize:
        feature_map = F.normalize(feature_map, dim=-1)
    features = feature_map.reshape(-1, feature_map.shape[-1]) # (H*W, C)
    reduction_mat = torch.pca_lowrank(features, q=q, niter=20)[2]
    colors = features @ reduction_mat
    # remove the first component
    colors_min = colors.min(dim=0).values
    colors_max = colors.max(dim=0).values
    tmp_colors = (colors - colors_min) / (colors_max - colors_min)
    fg_mask = tmp_colors[..., 0] < fg_mask_threshold
    fg_mask = fg_mask.reshape(feature_map.shape[:3])

    # T x H x 1 -> img_size
    fg_mask = F.interpolate(
        fg_mask.unsqueeze(0).float(),
        size=img_size,
        mode=interpolation,
    ).squeeze(0).cpu().numpy()
    return fg_mask

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@torch.no_grad()
def run(args):
    dino_embed_video = torch.load(args.dino_embed_video_path) # T x C x H x W
    res = (args.h, args.w)

    fg_mask = get_fg_mask_from_pca(feature_map=dino_embed_video.permute(0, 2, 3, 1), 
                                         img_size=res, interpolation="nearest", 
                                         q=args.q, fg_mask_threshold=args.fg_mask_threshold)
    fg_mask = (fg_mask * 255).astype(np.uint8)
    frames_path = save_video_frames(fg_mask, args.mask_path)
    print(f"Saved fg. mask to {frames_path}")    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino-embed-video-path", type=str, required=True)
    parser.add_argument("--h", type=int, required=True)
    parser.add_argument("--w", type=int, required=True)
    parser.add_argument("--mask-path", type=str, required=True)
    parser.add_argument("--fg_mask_threshold", type=float, default=0.4)
    parser.add_argument("--q", type=int, default=3)

    args = parser.parse_args()
    run(args)
