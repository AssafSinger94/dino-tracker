import argparse
import os
import torch
import yaml
from data.data_utils import load_video
from utils import add_config_paths, get_dino_features_video
device = "cuda" if torch.cuda.is_available() else "cpu"

def save_dino_embed_video(args):
    config_paths = add_config_paths(args.data_path, {})
    video_folder = config_paths["video_folder"]
    dino_embed_video_path = config_paths["dino_embed_video_path"] if not args.for_mask else config_paths["mask_dino_embed_video_path"]
    config = yaml.safe_load(open(args.config, "r").read())
    dino_model_name = config["dino_model_name"]if not args.for_mask else config["mask_dino_model_name"]
    dino_facet = config["dino_facet"]if not args.for_mask else config["mask_dino_facet"]
    dino_layer = config["dino_layer"]if not args.for_mask else config["mask_dino_layer"]
    dino_stride = config["dino_stride"]if not args.for_mask else config["mask_dino_stride"]
    h, w = config["video_resh"], config["video_resw"]
    
    video = load_video(video_folder=video_folder, resize=(h, w), num_frames=400).to(device) # T x 3 x H x W
    dino_embed_video = get_dino_features_video(video=video, model_name=dino_model_name, facet=dino_facet, stride=dino_stride, layer=dino_layer).to(device).detach() # T x C' x H' x W'

    os.makedirs(os.path.dirname(dino_embed_video_path), exist_ok=True)
    torch.save(dino_embed_video, dino_embed_video_path)
    print(f"Saved {dino_embed_video_path}, shape: {dino_embed_video.shape}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/preprocessing.yaml", type=str)
    parser.add_argument("--data-path", default="./dataset/tapvid-davis/davis_480/0", type=str, required=True)
    parser.add_argument("--for-mask", action="store_true", default=False)

    args = parser.parse_args()
    save_dino_embed_video(args)
