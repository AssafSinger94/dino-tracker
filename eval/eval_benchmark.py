import argparse
from tqdm import tqdm
import pandas as pd
import os
import pickle
from eval.metrics import compute_tapvid_metrics_for_video, compute_badja_metrics_for_video


def eval_dataset(args):
    benchmark_data = pickle.load(open(args.benchmark_pickle_path, "rb"))

    metrics_list = []

    dataset_root = args.dataset_root_dir
    for video_idx_str in tqdm(os.listdir(dataset_root), desc="Evaluating dataset"):
        if video_idx_str.startswith("."):
            continue
        video_dir = os.path.join(dataset_root, video_idx_str)
        trajectories_dir = os.path.join(video_dir, "trajectories")
        occlusions_dir = os.path.join(video_dir, "occlusions")
        video_idx = int(video_idx_str)

        if args.dataset_type == "tapvid":
            metrics = compute_tapvid_metrics_for_video(model_trajectories_dir=trajectories_dir, 
                                                        model_occ_pred_dir=occlusions_dir,
                                                        video_idx=video_idx,
                                                        benchmark_data=benchmark_data,
                                                        pred_video_sizes=[854, 476])
        elif args.dataset_type == "BADJA":
            metrics = compute_badja_metrics_for_video(model_trajectories_dir=trajectories_dir, 
                                                      video_idx=video_idx,
                                                      benchmark_data=benchmark_data,
                                                      pred_video_sizes=[854, 476])
        else:
            raise ValueError("Invalid dataset type. Must be either tapvid or BADJA")
        metrics["video_idx"] = int(video_idx)
        metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.set_index('video_idx', inplace=True)
    metrics_df.loc['average', :] = metrics_df.mean()
    metrics_df.to_csv(args.out_file)
    print("Total metrics:") 
    print(metrics_df.mean())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root-dir", default="./dataset/davis_256", type=str)
    parser.add_argument("--benchmark-pickle-path", default="./dataset/davis.pkl", type=str)
    parser.add_argument("--out-file", default="./tapvid/comp_metrics.csv", type=str)
    parser.add_argument("--dataset-type", default="tapvid", type=str, help="Dataset type: tapvid or BADJA")
    args = parser.parse_args()
    eval_dataset(args)
