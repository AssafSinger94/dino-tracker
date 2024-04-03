import os
import subprocess
import argparse
import torch
import yaml
from utils import add_config_paths

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/preprocessing.yaml", help="Config path", type=str)
    parser.add_argument("--data-path", default="./dataset/libby", type=str)

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f.read())
        config = add_config_paths(args.data_path, config)
    
    # 1. compute raft optical flows trajectories with direct flow filtering if specified, used for training
    args_list = ['python', './preprocessing/extract_trajectories.py', 
                    '--frames-path', config['video_folder'], '--output-path', config['trajectories_file'], '--min-trajectory-length', str(config['min_trajectory_length']),
                    '--threshold', str(config['threshold']), '--infer-res-size', str(config['video_resh']), str(config['video_resw'])]
    if config['filter_using_direct_flow']:
        args_list += ['--filter-using-direct-flow', '--direct-flow-threshold', str(config['direct_flow_threshold'])]
    print(f"----- Running {' '.join(args_list)}", flush=True)
    subprocess.run(args_list)

    # 2. compute DINO embeddings for training & dino-bb
    args_list = ['python', './preprocessing/save_dino_embed_video.py', 
                    '--data-path', args.data_path, '--config', args.config]
                    
    print(f"----- Running {' '.join(args_list)}", flush=True)
    subprocess.run(args_list)

    # 3. create FG masks using DINO features - if GT masks are not provided
    if not os.path.exists(config['masks_path']):
        # compute DINO embeddings for fg masks
        args_list = ['python', './preprocessing/save_dino_embed_video.py',
                        '--data-path', args.data_path, '--config', args.config, '--for-mask']
        print(f"----- Running {' '.join(args_list)}", flush=True)
        subprocess.run(args_list)

        args_list = ['python', './preprocessing/create_fg_mask.py', 
                     '--dino-embed-video-path', config['mask_dino_embed_video_path'], '--h', str(config['video_resh']), '--w', str(config['video_resw']), '--mask-path', config['masks_path'], 
                        '--fg_mask_threshold', str(config['fg_mask_threshold'])]
        print(f"----- Running {' '.join(args_list)}", flush=True)
        subprocess.run(args_list)
    else:
        print("Masks already exist, skipping...", flush=True)
    
    # 4. split trajectories to FG & BG
    args_list = ['python', './preprocessing/split_trajectories_to_fg_bg.py', 
                 '--traj_path', config['trajectories_file'], '--fg_masks_path', config['masks_path'], '--fg_traj_path', config['fg_trajectories_file'], '--bg_traj_path', config['bg_trajectories_file']]
    print(f"----- Running {' '.join(args_list)}", flush=True)
    subprocess.run(args_list)
    
    # 5. preprocess DINO best-buddies
    args_list = ['python', './preprocessing_dino_bb/main_dino_bb_preprocessing.py', 
                 '--config', args.config, '--data-path', str(args.data_path)]
    print(f"----- Running {' '.join(args_list)}", flush=True)
    subprocess.run(args_list)
