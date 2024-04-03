import os
import numpy as np
from typing import Mapping
from data.tapvid import get_video_config_by_video_id


def compute_tapvid_metrics(
    query_points: np.ndarray,
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
    query_mode: str,
    get_trackwise_metrics: bool = False,
) -> Mapping[str, np.ndarray]:
  """Computes TAP-Vid metrics (Jaccard, Pts. Within Thresh, Occ. Acc.)

  See the TAP-Vid paper for details on the metric computation.  All inputs are
  given in raster coordinates.  The first three arguments should be the direct
  outputs of the reader: the 'query_points', 'occluded', and 'target_points'.
  The paper metrics assume these are scaled relative to 256x256 images.
  pred_occluded and pred_tracks are your algorithm's predictions.

  This function takes a batch of inputs, and computes metrics separately for
  each video.  The metrics for the full benchmark are a simple mean of the
  metrics across the full set of videos.  These numbers are between 0 and 1,
  but the paper multiplies them by 100 to ease reading.

  Args:
     query_points: The query points, an in the format [t, y, x].  Its size is
       [b, n, 3], where b is the batch size and n is the number of queries
     gt_occluded: A boolean array of shape [b, n, t], where t is the number of
       frames.  True indicates that the point is occluded.
     gt_tracks: The target points, of shape [b, n, t, 2].  Each point is in the
       format [x, y]
     pred_occluded: A boolean array of predicted occlusions, in the same format
       as gt_occluded.
     pred_tracks: An array of track predictions from your algorithm, in the same
       format as gt_tracks.
     query_mode: Either 'first' or 'strided', depending on how queries are
       sampled.  If 'first', we assume the prior knowledge that all points
       before the query point are occluded, and these are removed from the
       evaluation.
     get_trackwise_metrics: if True, the metrics will be computed for every
       track (rather than every video, which is the default).  This means
       every output tensor will have an extra axis [batch, num_tracks] rather
       than simply (batch).

  Returns:
      A dict with the following keys:

      occlusion_accuracy: Accuracy at predicting occlusion.
      pts_within_{x} for x in [1, 2, 4, 8, 16]: Fraction of points
        predicted to be within the given pixel threshold, ignoring occlusion
        prediction.
      jaccard_{x} for x in [1, 2, 4, 8, 16]: Jaccard metric for the given
        threshold
      average_pts_within_thresh: average across pts_within_{x}
      average_jaccard: average across jaccard_{x}
  """

  summing_axis = (2,) if get_trackwise_metrics else (1, 2)

  metrics = {}

  eye = np.eye(gt_tracks.shape[2], dtype=np.int32)
  if query_mode == 'first':
    # evaluate frames after the query frame
    query_frame_to_eval_frames = np.cumsum(eye, axis=1) - eye
  elif query_mode == 'strided':
    # evaluate all frames except the query frame
    query_frame_to_eval_frames = 1 - eye
  else:
    raise ValueError('Unknown query mode ' + query_mode)

  query_frame = query_points[..., 0]
  query_frame = np.round(query_frame).astype(np.int32)
  evaluation_points = query_frame_to_eval_frames[query_frame] > 0

  # Occlusion accuracy is simply how often the predicted occlusion equals the
  # ground truth.
  occ_acc = np.sum(
      np.equal(pred_occluded, gt_occluded) & evaluation_points,
      axis=summing_axis,
  ) / np.sum(evaluation_points, axis=summing_axis)
  metrics['occlusion_accuracy'] = occ_acc

  # Next, convert the predictions and ground truth positions into pixel
  # coordinates.
  visible = np.logical_not(gt_occluded)
  pred_visible = np.logical_not(pred_occluded)
  all_frac_within = []
  all_jaccard = []
  for thresh in [1, 2, 4, 8, 16]:
    # True positives are points that are within the threshold and where both
    # the prediction and the ground truth are listed as visible.
    within_dist = np.sum(
        np.square(pred_tracks - gt_tracks),
        axis=-1,
    ) < np.square(thresh)
    is_correct = np.logical_and(within_dist, visible)

    # Compute the frac_within_threshold, which is the fraction of points
    # within the threshold among points that are visible in the ground truth,
    # ignoring whether they're predicted to be visible.
    count_correct = np.sum(
        is_correct & evaluation_points,
        axis=summing_axis,
    )
    count_visible_points = np.sum(
        visible & evaluation_points, axis=summing_axis
    )
    frac_correct = count_correct / count_visible_points
    metrics['pts_within_' + str(thresh)] = frac_correct
    all_frac_within.append(frac_correct)

    true_positives = np.sum(
        is_correct & pred_visible & evaluation_points, axis=summing_axis
    )

    # The denominator of the jaccard metric is the true positives plus
    # false positives plus false negatives.  However, note that true positives
    # plus false negatives is simply the number of points in the ground truth
    # which is easier to compute than trying to compute all three quantities.
    # Thus we just add the number of points in the ground truth to the number
    # of false positives.
    #
    # False positives are simply points that are predicted to be visible,
    # but the ground truth is not visible or too far from the prediction.
    gt_positives = np.sum(visible & evaluation_points, axis=summing_axis)
    false_positives = (~visible) & pred_visible
    false_positives = false_positives | ((~within_dist) & pred_visible)
    false_positives = np.sum(
        false_positives & evaluation_points, axis=summing_axis
    )
    jaccard = true_positives / (gt_positives + false_positives)
    metrics['jaccard_' + str(thresh)] = jaccard
    all_jaccard.append(jaccard)
  metrics['average_jaccard'] = np.mean(
      np.stack(all_jaccard, axis=1),
      axis=1,
  )
  metrics['average_pts_within_thresh'] = np.mean(
      np.stack(all_frac_within, axis=1),
      axis=1,
  )
  return metrics


def compute_tapvid_metrics_for_video(
        model_trajectories_dir: str, 
        model_occ_pred_dir: str,
        benchmark_data: dict,
        video_idx: int,
        pred_video_sizes=None,
    ):
    """Compute model metrics for TAP-Vid dataset. for a single video.
    Args:
        model_trajectories_dir (str): directory containing model trajectories.
        model_occ_pred_dir (str): directory containing model occlusion predictions.
        benchmark_data (dict): benchmark data dictionary.
        video_idx (int): video index.
        pred_video_sizes (Tuple[int, int]): predicted video sizes. Defaults to None.
    Returns:
        dict: computed metrics.
    """

    benchmark_video_data  = get_video_config_by_video_id(benchmark_data, video_idx)
    pred_rescale_h = benchmark_video_data['h'] if pred_video_sizes is None else pred_video_sizes[1]
    pred_rescale_w = benchmark_video_data['w'] if pred_video_sizes is None else pred_video_sizes[0]

    video_query_points_list = []
    gt_occluded_list = []
    gt_tracks_list = []
    pred_occluded_list = []
    pred_tracks_list = []
    
    for frame_idx in benchmark_video_data['query_points']:
        pred_tracks_path = os.path.join(model_trajectories_dir, f"trajectories_{frame_idx}.npy")
        pred_occluded_path =  os.path.join(model_occ_pred_dir, f"occlusion_preds_{frame_idx}.npy")

        assert os.path.exists(pred_tracks_path), f"failed to load {pred_tracks_path}"
        assert os.path.exists(pred_occluded_path), f"failed to load {pred_occluded_path}"
        
        trajectories = np.load(pred_tracks_path)
        pred_occluded = np.load(pred_occluded_path)

        query_points = np.array(benchmark_video_data['query_points'][frame_idx])
        t = np.array([frame_idx] * query_points.shape[0])
        query_points = np.concatenate([t[:, None], query_points], axis=1)

        video_query_points_list.append(query_points)
        gt_tracks_list.append(benchmark_video_data['target_points'][frame_idx])
        gt_occluded_list.append(benchmark_video_data['occluded'][frame_idx])
        pred_tracks_list.append(trajectories)
        pred_occluded_list.append(pred_occluded)
    
    video_query_points = np.concatenate(video_query_points_list, axis=0, dtype=np.float32) # N x 3
    gt_tracks = np.concatenate(gt_tracks_list, axis=0, dtype=np.float32) # N x T x 2
    gt_occluded = np.concatenate(gt_occluded_list, axis=0, dtype=object) # N x T
    pred_tracks = np.concatenate(pred_tracks_list, axis=0, dtype=np.float32) # N x T x 2
    pred_occluded = np.concatenate(pred_occluded_list, axis=0, dtype=object) # N x T

    # rescale and replace (t, x, y) with (t, y, x)
    video_query_points[..., 1] = video_query_points[..., 2] * 256 / (benchmark_video_data['h'])
    video_query_points[..., 2] = video_query_points[..., 1] * 256 / (benchmark_video_data['w'])
    
    gt_tracks[..., 0] *= 256 / (benchmark_video_data['w'])
    gt_tracks[..., 1] *= 256 / (benchmark_video_data['h'])
    
    pred_tracks[..., 0] *= 256 / pred_rescale_w
    pred_tracks[..., 1] *= 256 / pred_rescale_h

    # add batch dimension to each
    video_query_points = video_query_points[None, ...] # 1 x N x 3
    gt_tracks = gt_tracks[None, ...] # 1 x N x T x 2
    gt_occluded = gt_occluded[None, ...] # 1 x N x T
    pred_tracks = pred_tracks[None, ...] # 1 x N x T x 2
    pred_occluded = pred_occluded[None, ...] # 1 x N x T
    
    metrics = compute_tapvid_metrics(video_query_points, gt_occluded, gt_tracks, pred_occluded, pred_tracks, query_mode="strided")
    metrics_clean = {key: value.item() for key, value in metrics.items()} # extract values from numpy arrays
    return metrics_clean
  

def compute_badja_metrics_for_video(
      model_trajectories_dir,
      benchmark_data,
      video_idx,
      pred_video_sizes=None,
    ):
    """Compute model metrics for BADJA dataset (in TAP-Vid dormat). for a single video.
    Args:
        model_trajectories_dir (str): directory containing model trajectories.
        benchmark_data (dict): benchmark data dictionary.
        video_idx (int): video index.
        pred_video_sizes (Tuple[int, int]): predicted video sizes. Defaults to None.
    Returns:
        dict: computed metrics.
    """
    
    benchmark_video_data  = get_video_config_by_video_id(benchmark_data, video_idx)
    # model evaluated in benchmark resolution, scale predictions accordingly if necessary
    pred_rescale_h = 1 if pred_video_sizes is None else (benchmark_video_data['h'] / pred_video_sizes[1])
    pred_rescale_w = 1 if pred_video_sizes is None else (benchmark_video_data['w'] / pred_video_sizes[0])
    
    pred_tracks = []
    gt_occluded = []
    gt_tracks = []

    for frame_idx in benchmark_video_data['target_points']:
        assert os.path.exists(os.path.join(model_trajectories_dir, f'trajectories_{frame_idx}.npy')), f"failed to load {os.path.join(model_trajectories_dir, f'trajectories_{frame_idx}.npy')}"
        trajectories = np.load(os.path.join(model_trajectories_dir, f'trajectories_{frame_idx}.npy'))
        
        pred_tracks.append(trajectories) # N' x T x 2
        gt_tracks.append(benchmark_video_data['target_points'][frame_idx]) # N' x T x 2
        gt_occluded.append(benchmark_video_data['occluded'][frame_idx]) # N' x T

    # concat all trajectories from all frames
    pred_tracks = np.concatenate(pred_tracks, axis=0) # N x T x 2
    gt_tracks = np.concatenate(gt_tracks, axis=0) # N x T x 2
    gt_occluded = np.concatenate(gt_occluded, axis=0) # N x T
    segmentation_masks = benchmark_video_data['segmentations'] # T x H x W

    # scale to x & y according to pred_rescle
    pred_tracks[..., 0] *= pred_rescale_w
    pred_tracks[..., 1] *= pred_rescale_h

    # compute metrics - accuracy seg-based & 3px-based
    segmentation_masks = (segmentation_masks > 0).astype(np.float32) # T x H x W
    accs_seg = []
    accs_3px = []
    for i in range(gt_tracks.shape[0]):
        for t in range(1, segmentation_masks.shape[0]):
            area = np.sum(segmentation_masks[t])
            thr = 0.2 * np.sqrt(area) # 0.2 * sqrt(area)
            vis = (gt_occluded[i, t] == 0) # if not occluded
            if vis > 0:
                coord_e = pred_tracks[i, t]  # 2
                coord_g = gt_tracks[i, t]  # 2
                dist = np.sqrt(np.sum((coord_e - coord_g) ** 2))
                accs_seg.append((dist < thr).astype(np.float32))
                accs_3px.append((dist < 3.0).astype(np.float32))
        
    res_seg = np.mean(np.stack(accs_seg)) * 100.0
    res_3px = np.mean(np.stack(accs_3px)) * 100.0
    return {'acc_seg' : res_seg, 'acc_3px' : res_3px}