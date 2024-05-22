import torch
import torch.nn as nn
from einops import rearrange
import os
import gc
from pathlib import Path
from models.networks.tracker_head import TrackerHead
from models.networks.delta_dino import DeltaDINO
from models.utils import load_pre_trained_model
from data.dataset import RangeNormalizer
from utils import bilinear_interpolate_video


EPS = 1e-08


class Tracker(nn.Module):
    def __init__(
        self,
        video=None,
        ckpt_path="",
        dino_embed_path="",
        dino_patch_size=14,
        stride=7,
        device="cuda:0",
        
        cyc_n_frames=4,
        cyc_batch_size_per_frame=256,
        cyc_fg_points_ratio=0.7,
        cyc_thresh=4
        ):
        super().__init__()

        self.stride = stride
        self.dino_patch_size = dino_patch_size
        self.device = device
        self.refined_features = None
        self.dino_embed_path = dino_embed_path
        self.ckpt_path = ckpt_path
        self.cyc_n_frames = cyc_n_frames
        self.cyc_batch_size_per_frame = cyc_batch_size_per_frame
        self.cyc_fg_points_ratio = cyc_fg_points_ratio
        self.cyc_thresh = cyc_thresh
        
        self.video = video
        
        # DINO embed
        self.load_dino_embed_video()

        # Delta-DINO
        self.delta_dino = DeltaDINO(vit_stride=self.stride).to(device)

        # CNN-Refiner
        t, c, h, w = self.video.shape
        self.cmap_relu = nn.ReLU(inplace=True)
        self.tracker_head = TrackerHead(use_cnn_refiner=True,
                                        patch_size=dino_patch_size,
                                        step_h=stride,
                                        step_w=stride,
                                        video_h=h,
                                        video_w=w).to(device)
        self.range_normalizer = RangeNormalizer(shapes=(w, h, self.video.shape[0]))
    
    @torch.no_grad()
    def load_dino_embed_video(self):
        """
        video: T x 3 x H' x W'
        self.dino_embed_video: T x C x H x W
        """
        assert os.path.exists(self.dino_embed_path)
        self.dino_embed_video = torch.load(self.dino_embed_path, map_location=self.device)
  
    def get_dino_embed_video(self, frames_set_t):
        dino_emb = self.dino_embed_video[frames_set_t.to(self.dino_embed_video.device)] if frames_set_t.device != self.dino_embed_video.device else self.dino_embed_video[frames_set_t]
        return dino_emb
    
    def normalize_points_for_sampling(self, points):
        t, c, vid_h, vid_w = self.video.shape
        h = vid_h
        w = vid_w
        patch_size = self.dino_patch_size
        stride = self.stride
        
        last_coord_h =( (h - patch_size) // stride ) * stride + (patch_size / 2)
        last_coord_w =( (w - patch_size) // stride ) * stride + (patch_size / 2)
        ah = 2 / (last_coord_h - (patch_size / 2))
        aw = 2 / (last_coord_w - (patch_size / 2))
        bh = 1 - last_coord_h * 2 / ( last_coord_h - ( patch_size / 2 ))
        bw = 1 - last_coord_w * 2 / ( last_coord_w - ( patch_size / 2 ))
        
        a = torch.tensor([[aw, ah, 1]]).to(self.device)
        b = torch.tensor([[bw, bh, 0]]).to(self.device)
        normalized_points = a * points + b
        return normalized_points
    
    def sample_embeddings(self, embeddings, source_points):
        """embeddings: T x C x H x W. source_points: B x 3, where the last dimension is (x, y, t), x and y are in [-1, 1]"""
        t, c, h, w = embeddings.shape
        sampled_embeddings = bilinear_interpolate_video(video=rearrange(embeddings, "t c h w -> 1 c t h w"),
                                                               points=source_points,
                                                               h=h,
                                                               w=w,
                                                               t=t,
                                                               normalize_w=False,
                                                               normalize_h=False,
                                                               normalize_t=True)
        sampled_embeddings = sampled_embeddings.squeeze()
        if len(sampled_embeddings.shape) == 1:
            sampled_embeddings = sampled_embeddings.unsqueeze(1)
        sampled_embeddings = sampled_embeddings.permute(1,0)
        return sampled_embeddings

    def get_refined_embeddings(self, frames_set_t, return_raw_embeddings=False):
        frames_dino_embeddings = self.get_dino_embed_video(frames_set_t=frames_set_t)
        refiner_input_frames = self.video[frames_set_t]

        # compute residual_embeddings in batches of size 8
        batch_size = 8
        n_frames = frames_set_t.shape[0]
        residual_embeddings = torch.zeros_like(frames_dino_embeddings)
        for i in range(0, n_frames, batch_size):
            end_idx = min(i+batch_size, n_frames)
            residual_embeddings[i:end_idx] = self.delta_dino(refiner_input_frames[i:end_idx], frames_dino_embeddings[i:end_idx])

        refined_embeddings = frames_dino_embeddings + residual_embeddings

        if return_raw_embeddings:
            return refined_embeddings, residual_embeddings, frames_dino_embeddings
        return refined_embeddings, residual_embeddings
    
    def cache_refined_embeddings(self, move_dino_to_cpu=False):
        refined_features, _ = self.get_refined_embeddings(torch.arange(0, self.video.shape[0]))
        self.refined_features = refined_features
        if move_dino_to_cpu:
            self.dino_embed_video = self.dino_embed_video.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()
    
    def uncache_refined_embeddings(self, move_dino_to_gpu=False):
        self.refined_features = None
        if move_dino_to_gpu:
            self.dino_embed_video = self.dino_embed_video.to("cuda")
        torch.cuda.empty_cache()
        gc.collect()
    
    def save_weights(self, iter):
        torch.save(self.tracker_head.state_dict(), Path(self.ckpt_path) / f"tracker_head_{iter}.pt")
        torch.save(self.delta_dino.state_dict(), Path(self.ckpt_path) / f"delta_dino_{iter}.pt")
        
    def load_weights(self, iter):
        self.tracker_head = load_pre_trained_model(
            torch.load(os.path.join(self.ckpt_path, f"tracker_head_{iter}.pt")),
            self.tracker_head
        )
        self.delta_dino = load_pre_trained_model(
            torch.load(os.path.join(self.ckpt_path, f"delta_dino_{iter}.pt")),
            self.delta_dino
        )
    
    def get_corr_maps_for_frame_set(self, source_embeddings, frame_embeddings_set, target_frame_indices):
        corr_maps_set = torch.einsum("bc,nchw->bnhw", source_embeddings, frame_embeddings_set)
        corr_maps = corr_maps_set[torch.arange(source_embeddings.shape[0]), target_frame_indices.int(), :, :]
        
        embeddings_norm = frame_embeddings_set.norm(dim=1)
        target_embeddings_norm = embeddings_norm[target_frame_indices.int()]
        source_embeddings_norm = source_embeddings.norm(dim=1).unsqueeze(-1).unsqueeze(-1)
        corr_maps_norm = (source_embeddings_norm * target_embeddings_norm)
        corr_maps = corr_maps / torch.clamp(corr_maps_norm, min=EPS)
        corr_maps = rearrange(corr_maps, "b h w -> b 1 h w")
        
        return corr_maps
    
    def get_point_predictions_from_embeddings(self, source_embeddings, frame_embeddings_set, target_frame_indices):
        corr_maps = self.get_corr_maps_for_frame_set(source_embeddings, frame_embeddings_set, target_frame_indices)
        coords = self.tracker_head(self.cmap_relu(corr_maps))
        return coords
    
    def get_point_predictions(self, inp, frame_embeddings):
        source_points_unnormalized, source_frame_indices, target_frame_indices, _ = inp
        source_points = self.normalize_points_for_sampling(source_points_unnormalized)
        source_embeddings = self.sample_embeddings(frame_embeddings, torch.cat([ source_points[:, :-1], source_frame_indices[:, None] ], dim=1)) # B x C
        return self.get_point_predictions_from_embeddings(source_embeddings, frame_embeddings, target_frame_indices)
    
    @torch.no_grad()
    def get_cycle_consistent_coords(self, frames_set_t, fg_masks):
        source_selector = torch.randint(frames_set_t.shape[0], (self.cyc_n_frames,), device=frames_set_t.device)
        target_selector = torch.randint(frames_set_t.shape[0], (self.cyc_n_frames,), device=frames_set_t.device)
        
        # create 2D meshgrid for size fg_masks and join them to a single tensor of coordinates [x,y]
        h, w = fg_masks.shape[-2:]
        x = torch.arange(w, device=fg_masks.device).float()
        y = torch.arange(h, device=fg_masks.device).float()
        yy, xx = torch.meshgrid(y, x)
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        grid_coords = torch.stack([xx, yy], dim=-1)
        
        source_points = []
        target_points = []
        cycle_points = []
        cyc_source_frame_indices = []
        cyc_target_frame_indices = []
        source_times = []
        target_times = []
        
        BATCH_SIZE_PER_FRAME = self.cyc_batch_size_per_frame
        BATCH_SIZE_FG = int(BATCH_SIZE_PER_FRAME * self.cyc_fg_points_ratio)
        BATCH_SIZE_BG = BATCH_SIZE_PER_FRAME - BATCH_SIZE_FG
        
        for source_idx, target_idx in zip(source_selector, target_selector):
            source_t = frames_set_t[source_idx]
            target_t = frames_set_t[target_idx]

            frame_fg_mask = fg_masks[source_t] > 0
            frame_bg_mask = ~frame_fg_mask
            frame_coords_fg = grid_coords[frame_fg_mask.reshape(-1)]
            frame_coords_fg = frame_coords_fg[torch.randperm(frame_coords_fg.shape[0])[:BATCH_SIZE_FG]]
            frame_coords_bg = grid_coords[frame_bg_mask.reshape(-1)]
            frame_coords_bg = frame_coords_bg[torch.randperm(frame_coords_bg.shape[0])[:BATCH_SIZE_BG]]
            frame_coords = torch.cat([frame_coords_fg, frame_coords_bg], dim=0)
            
            frame_coords = torch.cat([frame_coords, torch.ones((frame_coords.shape[0], 1), device=frame_coords.device)*source_t], dim=-1)
            
            source_frame_indices = torch.tensor([source_idx]*frame_coords.shape[0], device=frames_set_t.device)
            target_frame_indices = torch.tensor([target_idx]*frame_coords.shape[0], device=frames_set_t.device)
            inp = frame_coords, source_frame_indices, target_frame_indices, frames_set_t
            
            # cycle-consistency filtering
            with torch.no_grad():
                target_coords = self.get_point_predictions(inp, self.frame_embeddings)                
                target_coords = self.range_normalizer.unnormalize(target_coords, src=(-1, 1), dims=[0, 1])
                target_coords = torch.cat([target_coords, torch.ones((target_coords.shape[0], 1), device=target_coords.device)*target_t], dim=-1)
                source_frame_indices = torch.tensor([target_idx]*target_coords.shape[0], device=frames_set_t.device)
                target_frame_indices = torch.tensor([source_idx]*target_coords.shape[0], device=frames_set_t.device)
                inp = target_coords, source_frame_indices, target_frame_indices, frames_set_t
                
                coords = self.get_point_predictions(inp, self.frame_embeddings)
                
                coords = self.range_normalizer.unnormalize(coords, src=(-1, 1), dims=[0, 1])
            filtered_source_indices = torch.norm(frame_coords[:, :2] - coords[:, :2], dim=1) <= self.cyc_thresh
            filtered_source_coords = frame_coords[filtered_source_indices]
            filtered_target_coords = target_coords[filtered_source_indices]
            filtered_cycle_coords = coords[filtered_source_indices]
            
            source_points.append(filtered_source_coords)
            target_points.append(filtered_target_coords)
            cycle_points.append(filtered_cycle_coords)
            cyc_source_frame_indices.append(torch.tensor([source_idx]*filtered_source_coords.shape[0], device=frames_set_t.device))
            cyc_target_frame_indices.append(torch.tensor([target_idx]*filtered_source_coords.shape[0], device=frames_set_t.device))  
            source_times.append(torch.tensor([source_t]*filtered_source_coords.shape[0], device=frames_set_t.device))
            target_times.append(torch.tensor([target_t]*filtered_source_coords.shape[0], device=frames_set_t.device))
        
        source_points = torch.cat(source_points, dim=0)
        target_points = torch.cat(target_points, dim=0)
        cycle_points = torch.cat(cycle_points, dim=0)
        cyc_source_frame_indices = torch.cat(cyc_source_frame_indices, dim=0)
        cyc_target_frame_indices = torch.cat(cyc_target_frame_indices, dim=0)
        source_times_normalized = self.range_normalizer(torch.cat(source_times, dim=0).unsqueeze(1).repeat(1, 3).float(), dst=(-1, 1), dims=[2])[:, 2]
        target_times_normalized = self.range_normalizer(torch.cat(target_times, dim=0).unsqueeze(1).repeat(1, 3).float(), dst=(-1, 1), dims=[2])[:, 2]
        
        return {
            "source_points": source_points,
            "target_points": target_points,
            "cycle_points": cycle_points,
            "source_frame_indices": cyc_source_frame_indices,
            "target_frame_indices": cyc_target_frame_indices,
            "source_times_normalized": source_times_normalized,
            "target_times_normalized": target_times_normalized,
        }
        
    def get_cycle_consistent_preds(self, frames_set_t, fg_masks):
        found_cycle_consistency_coords = False
        while not found_cycle_consistency_coords:
            cycle_consistency_coords =\
                self.get_cycle_consistent_coords(frames_set_t, fg_masks)
            found_cycle_consistency_coords = cycle_consistency_coords["source_points"].shape[0] > 0
        
        source_target_input = (cycle_consistency_coords["source_points"],
                                cycle_consistency_coords["source_frame_indices"],
                                cycle_consistency_coords["target_frame_indices"],
                                frames_set_t)
        target_source_input = (cycle_consistency_coords["target_points"],
                                cycle_consistency_coords["target_frame_indices"],
                                cycle_consistency_coords["source_frame_indices"],
                                frames_set_t)
        source_target_coords = self.get_point_predictions(source_target_input, self.frame_embeddings)
        target_source_coords = self.get_point_predictions(target_source_input, self.frame_embeddings)
        cycle_consistency_dists = torch.norm(cycle_consistency_coords["cycle_points"][:, :2] - cycle_consistency_coords["source_points"][:, :2], dim=1)
        
        cycle_source_points_normalized =\
            self.range_normalizer(cycle_consistency_coords["source_points"], dst=[-1, 1])
        cycle_target_points_normalized =\
            self.range_normalizer(cycle_consistency_coords["target_points"], dst=[-1, 1])
        cycle_consistency_preds = {
            "source_coords": cycle_source_points_normalized,
            "target_coords": cycle_target_points_normalized,
            "source_target_coords": source_target_coords[:, :2],
            "target_source_coords": target_source_coords[:, :2],
            "cycle_consistency_dists": cycle_consistency_dists,
            "cycle_points": cycle_consistency_coords["cycle_points"]
        }

        return cycle_consistency_preds

    def forward(self, inp, use_raw_features=False, cache_raw_features=False):
        """
        inp: source_points_unnormalized, source_frame_indices, target_frame_indices, frames_set_t; where
            source_points_unnormalized: B x 3. ((x, y, t) in image scale - NOT normalized)
            source_frame_indices: the indices of frames of source points in frames_set_t
            target_frame_indices: the indices of target frames in frames_set_t
            frames_set_t: N, 0 to T-1 (NOT normalized)
        use_raw_features: if True, use raw embeddings from DINO.
        cache_raw_features: if True, cache raw embeddings for future use.
        """
        frames_set_t = inp[-1]
        
        if use_raw_features:
            frame_embeddings = raw_embeddings = self.get_dino_embed_video(frames_set_t=frames_set_t)
        elif self.refined_features is not None: # load from cache
            frame_embeddings = self.refined_features[frames_set_t]
            if cache_raw_features:
                self.raw_embeddings = self.dino_embed_video[frames_set_t.to(self.dino_embed_video.device)]
        else:
            frame_embeddings, residual_embeddings, raw_embeddings = self.get_refined_embeddings(frames_set_t, return_raw_embeddings=True)
            self.residual_embeddings = residual_embeddings
            self.raw_embeddings = raw_embeddings
        self.frame_embeddings = frame_embeddings
        coords = self.get_point_predictions(inp, frame_embeddings)

        return coords
