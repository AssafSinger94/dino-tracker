from typing import Dict, List
import torch
from tqdm import tqdm
from data.dataset import RangeNormalizer
from models.tracker import Tracker

# ---- Functions for generating trajectories ----
def generate_trajectory_input(query_point, video, start_t=None, end_t=None):
    """
    Receives a single point (x,y,t) and the video, and generates input for Tracker model.
    Args:
        query_point: shape 3. (x,y,t).
        video: shape T x H x W x 3.
    Returns:
        source_points, source_frame_indices, target_frame_indices, frames_set_t.
        source_points: query_point repeated rest times. shape rest x 3. (x,y,t).
        source_frame_indices: [0] repeated rest times. shape rest x 1.
        target_frame_indices: 0 to rest-1. shape rest.
        frames_set_t: [query_point[0, 2], start_t, ..., end_t]. shape rest + 1.
    """
    start_t = 0 if start_t is None else start_t
    end_t = video.shape[0] if end_t is None else end_t
    video_subset = video[start_t:end_t]
    rest = video_subset.shape[0]
    device = video.device
    
    source_points = query_point.unsqueeze(0).repeat(rest, 1) # rest x 3

    frames_set_t = torch.arange(start_t, end_t, dtype=torch.long, device=device) # rest
    frames_set_t = torch.cat([ torch.tensor([query_point[2]], device=device), frames_set_t ]).int() # rest + 1
    source_frame_indices = torch.tensor([0], device=device).repeat(end_t-start_t) # rest
    target_frame_indices = torch.arange(rest, dtype=torch.long, device=device) + 1 # T
    
    return source_points, source_frame_indices, target_frame_indices, frames_set_t


@torch.no_grad()
def generate_trajectory(query_point:torch.tensor, video:torch.tensor, model:torch.nn.Module, range_normalizer:RangeNormalizer, dst_range=(-1, 1), use_raw_features=False,
                               batch_size=None) -> torch.tensor:
    """
    Genrates trajectory using tracker predictions for all timesteps.
    Returns:
        trajectory_pred: rest x 3. (x,y,t) coordinates for each timestep.
    """
    batch_size = video.shape[0] if batch_size is None else batch_size
    
    trajectory_pred = []
    for start_t in range(0, video.shape[0], batch_size):
        end_t = min(start_t + batch_size, video.shape[0])
        trajectory_input = generate_trajectory_input(query_point, video, start_t=start_t, end_t=end_t)
        trajectory_coordinate_preds_normalized = model(trajectory_input, use_raw_features=use_raw_features)
        trajectory_coordinate_preds = range_normalizer.unnormalize(trajectory_coordinate_preds_normalized, dims=[0,1], src=dst_range)
        trajectory_timesteps = trajectory_input[-1][1:].to(dtype=torch.float32) # rest
        trajectory_pred_cur = torch.cat([trajectory_coordinate_preds, trajectory_timesteps.unsqueeze(dim=1)], dim=1)
        trajectory_pred.append(trajectory_pred_cur)
    trajectory_pred = torch.cat(trajectory_pred, dim=0)
    return trajectory_pred

@torch.no_grad()
def generate_trajectories(query_points:torch.tensor, video:torch.tensor, model:torch.nn.Module, range_normalizer:RangeNormalizer, dst_range=(-1, 1), use_raw_features=False,
                                 batch_size=None) -> torch.tensor:
    """
    Genrates trajectories using tracker predictions. wraps generate_trajectory function.
    Returns:
        trajectories: len(query_points) x rest x 3. (x,y,t) coordinates for each trajectory.
    """
    trajectories_list = []
    query_points = query_points.to(dtype=torch.float32) # just in case
    for query_point in query_points:
        trajectory_pred = generate_trajectory(query_point=query_point, video=video, model=model, range_normalizer=range_normalizer, dst_range=dst_range, use_raw_features=use_raw_features,
                                                     batch_size=batch_size)
        trajectories_list.append(trajectory_pred)
    trajectories = torch.stack(trajectories_list)
    return trajectories



class ModelInference(torch.nn.Module):
    def __init__(
        self,
        model: Tracker,
        range_normalizer: RangeNormalizer,
        anchor_cosine_similarity_threshold: float = 0.5,
        cosine_similarity_threshold: float = 0.5,
        ) -> None:
        super().__init__()


        self.model = model
        self.model.eval()
        self.model.cache_refined_embeddings()

        self.range_normalizer = range_normalizer
        self.anchor_cosine_similarity_threshold = anchor_cosine_similarity_threshold
        self.cosine_similarity_threshold = cosine_similarity_threshold
    
    def compute_trajectories(self, query_points: torch.Tensor, batch_size=None,) -> torch.Tensor:
        trajecroies = generate_trajectories(
            query_points=query_points,
            model=self.model,
            video=self.model.video,
            range_normalizer=self.range_normalizer,
            dst_range=(-1,1),
            use_raw_features=False,
            batch_size=batch_size,
        )
        return trajecroies
    
    # ----------------- Cosine Similarity -----------------
    def compute_trajectory_cos_sims(self, trajectories, query_points) -> torch.Tensor:
        """Compute cosine similarities between trajectories and query points.
        Args:
            trajectories (torch.Tensor): Trajectories. N x T x 3. N is the number of trajectories. T is the number of time steps. (x, y, t).
            query_points (torch.Tensor): Query points. N x 3. used for retrieving corresponding query frames.
        Returns:
            trajectories_cosine_similarities (torch.Tensor): Cosine similarities between trajectories and query points. N x T."""
        # compute refined_features_at_trajectories
        N, T = trajectories.shape[:2]
        trajectories_normalized = self.model.normalize_points_for_sampling(trajectories) # N x T x 3
        refined_features_at_trajectories = self.model.sample_embeddings(self.model.refined_features, trajectories_normalized.view(-1, 3)) # (N*T) x C
        refined_features_at_trajectories = refined_features_at_trajectories.view(N, T, -1) # N x T x C
        
        query_frames = query_points[:, 2].long() # N
        refined_features_at_query_frames = refined_features_at_trajectories[torch.arange(N).to(self.model.device), query_frames] # N x C
        trajectories_cosine_similarities = torch.nn.functional.cosine_similarity(refined_features_at_query_frames.unsqueeze(1), refined_features_at_trajectories, dim=-1) # N x T
        return trajectories_cosine_similarities


    # ----------------- Anchor Trajectories (slower, but less memory-consuming) -----------------
    def _get_model_preds_at_anchors_old(self, model, range_normalizer, preds, anchor_indices, batch_size=None):
        """ preds: N"""
        batch_size = batch_size if batch_size is not None else preds.shape[0]
        
        cycle_coords = []
        # for each anchor frame in anchor_indices, get tracking predictions from all preds to the anchor frame
        for vis_frame in anchor_indices:
            # iterate over frames_set_t in batches of size batch_size
            coords = []
            for st_idx in range(0, preds.shape[0], batch_size):
                end_idx = min(st_idx + batch_size, preds.shape[0])
                frames_set_t = torch.arange(st_idx, end_idx, device=model.device) # source frames
                frames_set_t = torch.cat([ torch.tensor([vis_frame], device=model.device), frames_set_t ]).int() # add target frame (vis_frame)
                source_frame_indices = torch.arange(1, frames_set_t.shape[0], device=model.device)
                target_frame_indices = torch.tensor([0]*(frames_set_t.shape[0]-1), device=model.device)
                inp = preds[st_idx:end_idx], source_frame_indices, target_frame_indices, frames_set_t
                batch_coords = model(inp)
                batch_coords = range_normalizer.unnormalize(batch_coords, src=(-1, 1), dims=[0, 1])
                coords.append(batch_coords)
            coords = torch.cat(coords)
            
            cycle_coords.append(coords[:, :2]) # prediction of a target point to the top percentile
            
        cycle_coords = torch.stack(cycle_coords) # N_anchors x T x 2
        
        return cycle_coords
    
    # ----------------- Anchor Trajectories -----------------
    def _get_model_preds_at_anchors(self, model, range_normalizer, preds, anchor_indices, batch_size=None):
        """ preds: T. anchor_indices (N_anchors).
        Returns: cycle_coords, N_anchors x T x 2.
        """
        T = preds.shape[0]
        batch_size = batch_size if batch_size is not None else T
        
        frames_set_t = torch.arange(0, T).int() # [0, 1, 2, ..., T-1]
        source_frames = torch.arange(0, T, device=model.device)
        source_frame_indices = source_frames.repeat(anchor_indices.shape[0]) # T*N_anchors, [0, 1, ..., T-1, 0, 1, ..., T-1, ...]
        query_points = preds.repeat(anchor_indices.shape[0], 1) # (T*N_anchors) x 3
        target_frame_indices = anchor_indices.unsqueeze(1).repeat(1, source_frames.shape[0]).view(-1) # T*N_anchors, [anchor_indices[0], ..., anchor_indices[0], anchor_indices[1], ..., anchor_indices[1], ...]
        inp = query_points, source_frame_indices, target_frame_indices, frames_set_t
        cycle_coords = model(inp) # (T*N_anchors) x 2
        cycle_coords = range_normalizer.unnormalize(cycle_coords, src=(-1, 1), dims=[0, 1])
        cycle_coords = cycle_coords.view(anchor_indices.shape[0], T, 2) # N_anchors x T x 2
        return cycle_coords # N_anchors x T x 2
    
    def compute_anchor_trajectories(self, trajectories: torch.Tensor, cos_sims: torch.Tensor, batch_size=None) -> torch.Tensor:
        N, T = trajectories.shape[:2]
        eql_anchor_cyc_predictions = {}
            
        for qp_idx in tqdm(range(N), desc=f"Interating over query points"):
            preds = trajectories[qp_idx] # (T x 3)
            anchor_frames = torch.arange(T).to(self.model.device)[cos_sims[qp_idx] >= self.anchor_cosine_similarity_threshold] # T
            cycle_coords_eql_anchor = self._get_model_preds_at_anchors(self.model, self.range_normalizer, preds=preds, anchor_indices=anchor_frames, batch_size=batch_size)
            eql_anchor_cyc_predictions[qp_idx] = cycle_coords_eql_anchor
        return eql_anchor_cyc_predictions
    
    
    # ----------------- Occlusion -----------------
    def compute_occ_pred_for_qp(self, green_trajectories_qp: torch.tensor, source_trajectories_qp: torch.tensor, traj_cos_sim_qp: torch.tensor, anch_sim_th: float, cos_sim_th: float):
        visible_at_st_frame_qp = traj_cos_sim_qp >= anch_sim_th
        dists_from_source = torch.norm(green_trajectories_qp - source_trajectories_qp[visible_at_st_frame_qp, :].unsqueeze(1), dim=-1)  # dists_from_source (M x T), dists_from_source[anchor_t, source_t] = dist

        anchor_median_errors = torch.median(dists_from_source[:, visible_at_st_frame_qp], dim=0).values  # T_vis
        median_anchor_dist_th = anchor_median_errors.max()  # float
        dists_from_source_anchor_vis = dists_from_source  # (T_vis x T)
        median_dists_from_source_anchor_vis = torch.median(dists_from_source_anchor_vis, dim=0).values  # T
        return ((median_dists_from_source_anchor_vis > median_anchor_dist_th) | (traj_cos_sim_qp < cos_sim_th))

    def compute_occlusion(self, trajectories: torch.Tensor, trajs_cos_sims: torch.Tensor, anchor_trajectories: Dict[int, torch.Tensor]) -> torch.Tensor:
        """Compute occlusion for trajectories.
        Args:
            trajectories (torch.Tensor): Trajectories. N x T x 3. N is the number of trajectories. T is the number of time steps. trajectory for qp_idx in query_points (N).
            trajs_cos_sims (torch.Tensor): Cosine similarities between trajectories and query points. N x T. traj_cos_sims[qp_idx, t] = cos_sim
            anchor_trajectories dict(torch.Tensor): Anchor trajectories. {qp_idx: T x T x 2}. N is the number of trajectories.
        Returns:
            occ_preds (torch.Tensor): Occlusion predictions. N x T. occ_preds[qp_idx, t] = 1 if occluded, 0 otherwise.
        """

        N = trajectories.shape[0]
        occ_preds_by_dist_th_anchor_frame_vis = []

        for qp_idx in range(N):
            source_trajectories_qp = trajectories[qp_idx, :, :2] # source_trajectories_qp (T x 2)
            traj_cos_sim_qp = trajs_cos_sims[qp_idx] # cos_sim_qp (T)
            green_trajectories_qp = anchor_trajectories[qp_idx] # (T x T x 2), green_trajectories_qp[achor_t, source_t] = [x, y], source_t = start_frame
            occ_preds_by_dist_th_anchor_frame_vis.append(self.compute_occ_pred_for_qp(green_trajectories_qp, source_trajectories_qp, traj_cos_sim_qp, self.anchor_cosine_similarity_threshold, self.cosine_similarity_threshold))

        occ_preds = torch.stack(occ_preds_by_dist_th_anchor_frame_vis) # (N x T)

        return occ_preds
    
    # ----------------- Inference -----------------
    @torch.no_grad()
    def infer(self, query_points: torch.Tensor, batch_size=None) -> torch.Tensor:
        """Infer trajectory and occlusion for query points.
        Args:
            query_points (torch.Tensor): Query points. N x 3. N is the number of query points. (x, y, t).
            batch_size (int): Batch size for inference. if None, all frames are inferred at once.
        Returns:
            trajectories (torch.Tensor): Predicted trajectory. N x T x 2. T is the number of time steps.
            occlusion (torch.Tensor): Predicted occlusion. N x T. T is the number of time steps."""
        trajs = self.compute_trajectories(query_points, batch_size) # N x T x 3
        cos_sims = self.compute_trajectory_cos_sims(trajs, query_points)
        anchor_trajs = self.compute_anchor_trajectories(trajs, cos_sims, batch_size)
        occ = self.compute_occlusion(trajs, cos_sims, anchor_trajs)
        return trajs[..., :2], occ # N x T x 2, N x T

