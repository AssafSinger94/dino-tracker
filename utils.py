import gc
import torch
from einops import rearrange
import torchvision.transforms as T

from models.extractor import VitExtractor
import os


def add_config_paths(data_path, config):
    # preprocessing
    config['video_folder'] = os.path.join(data_path, "video")
    config['trajectories_file'] = os.path.join(data_path, "of_trajectories", "trajectories.pt")
    config['unfiltered_trajectories_file'] = os.path.join(data_path, "of_trajectories", "trajectories_wo_direct_filter.pt")
    config['fg_trajectories_file'] = os.path.join(data_path, "of_trajectories", "fg_trajectories.pt")
    config['bg_trajectories_file'] = os.path.join(data_path, "of_trajectories", "bg_trajectories.pt")
    config['dino_embed_video_path'] = os.path.join(data_path, "dino_embeddings", "dino_embed_video.pt")
    config['dino_bb_dir'] = os.path.join(data_path, "dino_best_buddies")
    config['mask_dino_embed_video_path'] = os.path.join(data_path, "dino_embeddings", "dino_embed_video-layer=23.pt")
    config['masks_path'] = os.path.join(data_path, "masks")
    # model
    config['ckpt_folder'] = os.path.join(data_path, "models", "dino_tracker")
    # outpts
    config['trajectories_dir'] = os.path.join(data_path, "trajectories")
    config['occlusions_dir'] = os.path.join(data_path, "occlusions")
    config['grid_trajectories_dir'] = os.path.join(data_path, "grid_trajectories")
    config['grid_occlusions_dir'] = os.path.join(data_path, "grid_occlusions")
    config['model_vis_dir'] = os.path.join(data_path, "visualizations")
    return config


@torch.no_grad()
def get_dino_features_video(video, model_name="dinov2_vitb14", facet='tokens', stride=7, layer=None, device: str = 'cuda:0'):
    """
    Args:
        video (torch.tensor): Tensor of the input video, of shape: T x 3 x H x W.
            T- number of frames. C- number of RGB channels (most likely 3), W- width, H- height.
        device (str, optional):indicating device type. Defaults to 'cuda:0'.

    Returns:
        dino_keys_video: DINO keys from last layer for each frame. Shape: (T x C x H//8 x W//8).
            T- number of frames. C - DINO key embedding dimension for patch.
    """
    dino_extractor = VitExtractor(model_name=model_name, device=device, stride=stride)
    dino_extractor = dino_extractor.eval().to(device)
    imagenet_norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ph = dino_extractor.get_height_patch_num(video[[0]].shape)
    pw = dino_extractor.get_width_patch_num(video[[0]].shape)
    dino_embedding_dim = dino_extractor.get_embedding_dim(model_name)
    n_layers = dino_extractor.get_n_layers()
    layers = [n_layers - 1] if layer is None else [layer]

    dino_features_video = torch.zeros(size=(video.shape[0], dino_embedding_dim, ph, pw), device='cpu')
    for i in range(video.shape[0]):
        dino_input = imagenet_norm(video[[i]]).to(device)
        if facet == "keys":
            features = dino_extractor.get_keys_from_input(dino_input, layers=layers)
        elif facet == "queries":
            features = dino_extractor.get_queries_from_input(dino_input, layers=layers)
        elif facet == "values":
            features = dino_extractor.get_values_from_input(dino_input, layers=layers)
        elif facet == "tokens":
            features = dino_extractor.get_feature_from_input(dino_input, layers=layers) # T (HxW + 1) x C
        else:
            raise ValueError(f"facet {facet} not supported")
        features = rearrange(features[:, 1:, :], "heads (ph pw) ch -> (ch heads) ph pw", ph=ph, pw=pw)
        dino_features_video[i] = features.cpu()
    # interpolate to the original video length
    del dino_extractor
    torch.cuda.empty_cache()
    gc.collect()
    return dino_features_video


def bilinear_interpolate_video(video:torch.tensor, points:torch.tensor, h:int, w:int, t:int, normalize_h=False, normalize_w=False, normalize_t=True):
    """
    Sample embeddings from an embeddings volume at specific points, using bilear interpolation per timestep.

    Args:
        video (torch.tensor): a volume of embeddings/features previously extracted from an image. shape: 1 x C x T x H' x W'
            Most likely used for DINO embeddings 1 x C x T x H' x W' (C=DINO_embeddings_dim, W'= W//8 & H'=H//8 of original image).
        points (torch.tensor): batch of B points (pixel cooridnates) (x,y,t) you wish to sample. shape: B x 3.
        h (int): True Height of images (as in the points) - H.
        w (int): Width of images (as in the points) - W.
        t (int): number of frames - T.

    Returns:
        sampled_embeddings: sampled embeddings at specific posiitons. shape: 1 x C x 1 x B x 1.
    """
    samples = points[None, None, :, None].detach().clone() # expand shape B x 3 TO (1 x 1 x B x 1 x 3), we clone to avoid altering the original points tensor.     
    if normalize_w:
        samples[:, :, :, :, 0] = samples[:, :, :, :, 0] / (w - 1)  # normalize to [0,1]
        samples[:, :, :, :, 0] = samples[:, :, :, :, 0] * 2 - 1  # normalize to [-1,1]
    if normalize_h:
        samples[:, :, :, :, 1] = samples[:, :, :, :, 1] / (h - 1)  # normalize to [0,1]
        samples[:, :, :, :, 1] = samples[:, :, :, :, 1] * 2 - 1  # normalize to [-1,1]
    if normalize_t:
        if t > 1:
            samples[:, :, :, :, 2] = samples[:, :, :, :, 2] / (t - 1)  # normalize to [0,1]
        samples[:, :, :, :, 2] = samples[:, :, :, :, 2] * 2 - 1  # normalize to [-1,1]
    return torch.nn.functional.grid_sample(video, samples, align_corners=True, padding_mode ='border') # points out-of bounds are padded with border values
