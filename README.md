# DINO-Tracker: Taming DINO for Self-Supervised Point Tracking in a Single Video (ECCV 2024)

## [<a href="https://dino-tracker.github.io/" target="_blank">Project Page</a>] [<a href="https://arxiv.org/abs/2403.14548" target="_blank">arXiv</a>]



https://github.com/AssafSinger94/dino-tracker/assets/43016459/bcd7d57a-1977-4f05-a2f4-ddd859524598




## Usage

1. [Setup](#setup)
2. [Preprocessing](#preprocessing)
3. [Training](#training)
4. [Inference](#inference)


## Setup

Clone the repository:

```git clone https://github.com/AssafSinger94/dino-tracker.git```

Switch to the project directory:

```cd dino-tracker```

To setup the environment, run:

```
conda create -n dino-tracker python=3.9
conda activate dino-tracker
pip install -r requirements.txt
```

Add current path to ```PYTHONPATH```:

```export PYTHONPATH=`pwd`:$PYTHONPATH```

## Preprocessing

Given an input video, we start by extracting optical flow and DINO best-buddy correspondences.
The input video directory should have the following structure:

```
├──<VIDEO_DIR>
    ├──video/
        ├──00000.png
        ├──00001.png
        ├──...
    ├──masks/ # optional
        ├──00000.png
        ├──00001.png
        ├──...
```

where `masks` contains the per-frame foreground masks. If `masks` is not provided, foreground masks are automatically computed using DINO features saliency maps.

In case the video is in mp4 format, convert it to frames by simply running:
```
python ./preprocessing/mp4_to_frames.py \
    --video-path <PATH_TO_MP4> \
    --output-folder <VIDEO_DIR_PATH>/video
```

To run the preprocessing pipeline, run the following:
```
python ./preprocessing/main_preprocessing.py \
    --config ./config/preprocessing.yaml \
    --data-path <VIDEO_DIR_PATH>
```

The script outputs chained optical flow trajectories, DINO embeddings and DINO best-buddies in the following structure:

```
├──<VIDEO_DIR>
    ├──video/
    ├──masks/
    ├──dino_best_buddies/
    ├──dino_embeddings/
    ├──of_trajectories/
```


## Training

Once preprocessing is finished, run the following command to train DINO-Tracker:
```
python ./train.py \
    --config ./config/train.yaml \
    --data-path <VIDEO_DIR_PATH>
```

The checkpoints are saved under:
```
├──<VIDEO_DIR>
    ├──models
        ├──dino_tracker
            ├──delta_dino_<ITER>.pt
            ├──tracker_head_<ITER>.pt
```


## Inference

### Trajectory creation and visualization

To predict and visualize trajectories with a trained DINO-Tracker, run the following scripts sequentially:


```
python ./inference_grid.py \
    --config ./config/train.yaml \
    --data-path <VIDEO_DIR_PATH> \
    --use-segm-mask # optional, used for sampling only foreground points
```


```
python visualization/visualize_rainbow.py \
    --data-path <VIDEO_DIR_PATH> \
    --plot-trails # optional, used for visualizing motion trails.
```

The first script creates trajectories for a grid of query points in the first frame, while the second script visualizes them. The `--plot-trails` option is used for visualizing motion trails. Note that this option requires a segmentation mask for the first frame. If `--plot-trails` is not provided, the script only visualizes the tracked positions in circles. The visualizations are outputted under `<VIDEO_DIR_PATH>/visualizations` directory.


### TAP-Vid evaluation

To evaluate on TAP-Vid-DAVIS, please see the following steps. The same steps can be applied for TAP-Vid Kinetics and BADJA datasets.


1. Download benchmark data file `tapvid_davis_data_strided.pkl` from [this link](https://www.dropbox.com/scl/fo/7s2rgsm92qbzzh2xnx51d/AIvXxRaJPL2RQm43Zi_taJU?rlkey=6cs0bm2u0on1u7z0jyxlq8avq&dl=0), put it under `./tapvid/tapvid_davis_data_strided.pkl`,

2. Download pre-trained weights and videos from [this link](https://www.dropbox.com/scl/fo/7s2rgsm92qbzzh2xnx51d/AIvXxRaJPL2RQm43Zi_taJU?rlkey=6cs0bm2u0on1u7z0jyxlq8avq&dl=0) under `davis_480.zip`, unzip the folder to `./dataset/tapvid-davis/`,

3. Extract DINO embeddings for all videos by running the following:
```
python ./preprocessing/save_dino_embed_video.py \
    --config ./config/preprocessing.yaml \
    --data-path ./dataset/tapvid-davis/<VIDEO_ID>
```
The above should be run for all videos in the benchmark, e.g. `<VIDEO_ID> = {0, 1, ..., 29}` for DAVIS.

4. Predict trajectories on benchmark query points by running the following for all benchmark videos:
```
python inference_benchmark.py \
    --config ./config/train.yaml \
    --data-path ./dataset/tapvid-davis/<VIDEO_ID> \
    --benchmark-pickle-path ./tapvid/tapvid_davis_data_strided.pkl \
    --video-id <VIDEO_ID>
```

5. Evaluate the model accuracy by running the following:
```
python ./eval/eval_benchmark.py \
    --dataset-root-dir ./dataset/tapvid-davis \
    --benchmark-pickle-path ./tapvid/tapvid_davis_data_strided.pkl \
    --out-file ./tapvid/comp_metrics_davis.csv \
    --dataset-type tapvid # tapvid | BADJA
```
The evaluation should output: 
```average_pts_within_thresh: 0.8066 | occlusion_acc: 0.8854 | average_jaccard: 0.6528```.

The output CSV file contains all TAP-Vid metrics (position accuracy, occlusion accuracy, Average Jaccard) for all videos.


## Citation
```
@misc{dino_tracker_2024,
    author        = {Tumanyan, Narek and Singer, Assaf and Bagon, Shai and Dekel, Tali},
    title         = {DINO-Tracker: Taming DINO for Self-Supervised Point Tracking in a Single Video},
    year          = {2024},
    booktitle = {European Conference on Computer Vision (ECCV)}
}
```
