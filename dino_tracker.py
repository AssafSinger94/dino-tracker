import logging
import os
import torch
import yaml
from pathlib import Path
from tqdm import tqdm
from einops import rearrange
from PIL import Image
from models.utils import filter_bb_foreground_pairs, get_last_ckpt_iter, get_feature_cos_sims, get_vit_feature_coords_from_mask
from models.tracker import Tracker
from optimization.schedulers import get_cnn_refiner_scheduler
from data.data_utils import load_video
from data.dataset import DinoTrackerSampler, RangeNormalizer
from preprocessing.split_trajectories_to_fg_bg import load_masks
from utils import add_config_paths


device = "cuda:0" if torch.cuda.is_available() else "cpu"



class DINOTracker():
    def __init__(self, args):
        
        self.load_config(args.config)
        self.set_paths(args.data_path)

        self.orig_video_res_h, self.orig_video_res_w, video_rest = self.get_original_video_res(self.video_path)
        self.range_normalizer = RangeNormalizer(shapes=(self.config["video_resw"], self.config["video_resh"], video_rest)).to(device) # nn.Module
        self.of_loss_fn = torch.nn.HuberLoss(delta=1/32, reduction='none')
        
    def load_fg_masks(self):
        self.fg_masks = torch.from_numpy(load_masks(self.fg_masks_path, h_resize=self.config["video_resh"])).to(device)        
    
    def set_paths(self, data_path):
        config_paths = add_config_paths(data_path, {})
        self.video_path = config_paths["video_folder"]
        self.fg_masks_path = config_paths["masks_path"]
        self.dino_embed_path = config_paths["dino_embed_video_path"]
        self.fg_trajectories_path = config_paths["fg_trajectories_file"]
        self.bg_trajectories_path = config_paths["bg_trajectories_file"]
        self.dino_bb_path = os.path.join(config_paths["dino_bb_dir"], "dino_best_buddies_filtered.pt")
        self.ckpt_folder = config_paths["ckpt_folder"]
        self.trajectories_dir = config_paths['trajectories_dir']
        self.occlusions_dir = config_paths['occlusions_dir']
        self.grid_trajectories_dir = config_paths['grid_trajectories_dir']
        self.grid_occlusions_dir = config_paths['grid_occlusions_dir']
        os.makedirs(self.ckpt_folder, exist_ok=True)
        
    def load_config(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f.read())


    def get_original_video_res(self, video_path):
        video_frames_list = sorted(list(Path(video_path).glob("*.jpg")) + list(Path(video_path).glob("*.png")))
        video_rest = len(video_frames_list)
        # read first frame using PIL.Image to get resolution
        frame = Image.open(video_frames_list[0])
        video_res_hw = frame.size[::-1]
        return video_res_hw + (video_rest,)

    def load_trajectories(self):
        assert os.path.exists(self.fg_trajectories_path) & os.path.exists(self.bg_trajectories_path), "trajectory files don't exist"
        
        trj_device = torch.device('cpu') if self.config['keep_traj_in_cpu'] else device
        train_fg_trajectories = torch.load(self.fg_trajectories_path, map_location=trj_device)
        train_bg_trajectories = torch.load(self.bg_trajectories_path, map_location=trj_device)
        return train_fg_trajectories, train_bg_trajectories
    
    def load_dino_best_buddies(self):
        self.dino_bb_pairs = torch.load(self.dino_bb_path, map_location=device)
    
    def get_sampler(self):        
        train_fg_trajectories, train_bg_trajectories = self.load_trajectories()
        train_sampler = DinoTrackerSampler(fg_trajectories=train_fg_trajectories,
                                           bg_trajectories=train_bg_trajectories,
                                           fg_traj_ratio=self.config["fg_traj_ratio"],
                                           batch_size=self.config["train_batch_size"],
                                           range_normalizer=self.range_normalizer,
                                           dst_range=(-1, 1),
                                           num_frames=self.config["batch_n_frames"],
                                           keep_in_cpu=self.config['keep_traj_in_cpu'])
        return train_sampler
    
    def get_model(self):
        video = load_video(video_folder=self.video_path, resize=(self.config["video_resh"], self.config["video_resw"])).to(device)
        tracker_args = {
            "video":video,
            "device":device,
            "dino_embed_path": self.dino_embed_path,
            "dino_patch_size": self.config["dino_patch_size"],
            "stride":self.config["stride"],
            "ckpt_path": self.ckpt_folder,
            
            "cyc_n_frames": self.config["cyc_n_frames"],
            "cyc_batch_size_per_frame": self.config["cyc_batch_size_per_frame"],
            "cyc_fg_points_ratio": self.config["cyc_fg_points_ratio"],
            "cyc_thresh": self.config["cyc_thresh"]
        }
        
        model = Tracker(**tracker_args).to(device)
            
        self.init_iter = get_last_ckpt_iter(self.ckpt_folder)
        if self.init_iter > 0:
            model.load_weights(self.init_iter)
        
        return model
    
    def train_setup(self):
        model = self.get_model()
        params = [{"params": model.delta_dino.parameters(), "lr": self.config["lr_delta_dino"]},
                  {"params": model.tracker_head.parameters(), "lr": self.config["lr_cnn_refiner"]}]
        optimizer = torch.optim.Adam(params)                    
        scheduler = get_cnn_refiner_scheduler(optimizer, gamma=self.config['scheduler_gamma'], apply_every=self.config['apply_scheduler_every'])
        
        if self.init_iter > 0:
            self.init_scheduler(scheduler, self.init_iter)
        print("------- INIT ITER", self.init_iter)        
        
        return model, optimizer, scheduler
    
    def init_scheduler(self, scheduler, iter):
        for i in range(iter):
            scheduler.step()
    
    def get_inputs_and_labels(self, sampler):
        sample = sampler()
        labels = sample["t2_points_normalized"][:, :-1]
        inputs = (sample["t1_points"], sample["source_frame_indices"], sample["target_frame_indices"], sample["frames_set_t"])
        return inputs, labels
    
    def set_model_train(self, model):
        model.train()

    def get_emb_norm_regularization_loss(self, model):
        refined_emb_norm = model.frame_embeddings.norm(dim=1) # b x h x w
        dino_emb_norm = model.raw_embeddings.norm(dim=1) # b x h x w
        norm_ratio = refined_emb_norm / dino_emb_norm
        return (norm_ratio - 1).abs().mean()
    
    def get_emb_angle_regularization_loss(self, model):
        refined_emb = model.frame_embeddings
        dino_emb = model.raw_embeddings
        cos_sims = get_feature_cos_sims(refined_emb, dino_emb)
        return (cos_sims - 1).abs().mean()
    
    def get_refiner_contrastive_loss(self, model, frames_set_t):
        contrastive_loss = self.get_refined_bb_contrastive_loss(model,
                                                                frames_set_t,
                                                                model.frame_embeddings,
                                                                batch_size=self.config["cl_n_frames"],
                                                                points_per_pair=self.config["cl_points_per_pair"],
                                                                fg_points_ratio=self.config["cl_fg_points_ratio"],
                                                                temp=self.config["cl_temp"],
                                                                cl_div=self.config["cl_div_ref_bb"])
        return contrastive_loss
    
    def get_dino_bb_contrastive_loss(self, model, frames_set_t):
        batch_size = self.config["cl_n_frames"]
        source_selector = torch.randint(frames_set_t.shape[0], (batch_size,), device=frames_set_t.device) # N
        target_selector = torch.randint(frames_set_t.shape[0], (batch_size,), device=frames_set_t.device) # N
        while (source_selector == target_selector).any():
            target_selector = torch.randint(frames_set_t.shape[0], (batch_size,), device=frames_set_t.device) # N
        
        BATCH_SIZE_PER_FRAME = self.config["cl_points_per_pair"]
        BATCH_SIZE_FG = int(BATCH_SIZE_PER_FRAME * self.config["cl_fg_points_ratio"])
        BATCH_SIZE_BG = BATCH_SIZE_PER_FRAME - BATCH_SIZE_FG
        
        n_total_bb = 0
        loss_contrastive_total = 0
        loss_cl1 = []
        loss_cl2 = []
        l_ws = []
        l_cos_ws = []
        for source_frame_idx, target_frame_idx in zip(source_selector, target_selector):
            if source_frame_idx == target_frame_idx:
                continue
            
            source_frame = frames_set_t[source_frame_idx]
            target_frame = frames_set_t[target_frame_idx]
            best_buddies = self.dino_bb_pairs[f'{source_frame}_{target_frame}']
            if best_buddies['source_coords'] is None or best_buddies['source_coords'].shape[0] == 0:
                continue

            source_coords_fg, target_coords_fg, fg_mask_source = filter_bb_foreground_pairs(best_buddies['source_coords'],
                                                                                            best_buddies['target_coords'],
                                                                                            self.fg_masks[source_frame],
                                                                                            resw=model.video.shape[-1],
                                                                                            resh=model.video.shape[-2])
            bg_mask_source = ~fg_mask_source
            source_coords_bg, target_coords_bg = best_buddies['source_coords'][bg_mask_source], best_buddies['target_coords'][bg_mask_source]
            fg_indices = torch.arange(best_buddies['source_coords'].shape[0], device=device)[fg_mask_source]
            bg_indices = torch.arange(best_buddies['source_coords'].shape[0], device=device)[bg_mask_source]
            
            source_coords_fg = torch.cat([source_coords_fg, torch.tensor([source_frame_idx] * source_coords_fg.shape[0]).unsqueeze(1).cuda()], dim=1)
            target_coords_fg = torch.cat([target_coords_fg, torch.tensor([target_frame_idx] * target_coords_fg.shape[0]).unsqueeze(1).cuda()], dim=1)
            fg_selector = torch.randperm(source_coords_fg.shape[0])[:BATCH_SIZE_FG]
            source_coords_fg_sampled = source_coords_fg[fg_selector]
            target_coords_fg_sampled = target_coords_fg[fg_selector]
            
            source_coords_bg = torch.cat([source_coords_bg, torch.tensor([source_frame_idx] * source_coords_bg.shape[0]).unsqueeze(1).cuda()], dim=1)
            target_coords_bg = torch.cat([target_coords_bg, torch.tensor([target_frame_idx] * target_coords_bg.shape[0]).unsqueeze(1).cuda()], dim=1)
            bg_selector = torch.randperm(source_coords_bg.shape[0])[:BATCH_SIZE_BG]
            source_coords_bg_sampled = source_coords_bg[bg_selector]
            target_coords_bg_sampled = target_coords_bg[bg_selector]
            selector = torch.cat([fg_indices[fg_selector], bg_indices[bg_selector]])
            
            source_points = torch.cat([ source_coords_fg_sampled, source_coords_bg_sampled ], dim=0)
            target_points = torch.cat([ target_coords_fg_sampled, target_coords_bg_sampled ], dim=0)
            
            if source_points.shape[0] == 0:
                continue
            
            source_coordinates_normalized = model.normalize_points_for_sampling(source_points) # B x 2
            target_coordinates_normalized = model.normalize_points_for_sampling(target_points) # B x 2
            source_bb_f = model.sample_embeddings(model.frame_embeddings,
                                                  source_coordinates_normalized) # B x C
            target_bb_f = model.sample_embeddings(model.frame_embeddings,
                                                  target_coordinates_normalized) # B x C
            source_f = rearrange(model.frame_embeddings[source_frame_idx], 'c h w -> (h w) c') # n x c, n = hxw
            target_f = rearrange(model.frame_embeddings[target_frame_idx], 'c h w -> (h w) c') # n x c, n = hxw
            cl1, cl2, _, _ = self.get_bb_pairs_contrastive_loss(source_bb_f, target_bb_f, source_f, target_f, temp=self.config['cl_temp'])
            n_total_bb += 2 * cl1.shape[0]
            loss_cl1.append(cl1)
            loss_cl2.append(cl2)
            
            ws = 1 - best_buddies['r'][selector]
            ws = torch.sigmoid(self.config["bb_amb_sig_a"] * ws + self.config["bb_amb_sig_b"])
            l_ws.append(ws)
            l_cos_ws.append(torch.clamp(2*(best_buddies['cos_sims'][selector]**3), 0))

        if n_total_bb == 0:
            return torch.tensor(0.).to(frames_set_t.device)
        
        l_ws = torch.cat(l_ws) # B
        l_cos_ws = torch.cat(l_cos_ws) # B
        loss_cl1 = torch.cat(loss_cl1) # B
        loss_cl2 = torch.cat(loss_cl2) # B

        cl_div = self.config["cl_div_dino_bb"]
        loss_contrastive_total = ( (loss_cl1 * l_ws * l_cos_ws / cl_div).sum() + (loss_cl2 * l_ws * l_cos_ws / cl_div).sum() ) / 2
        return loss_contrastive_total
    
    def get_refined_bb_contrastive_loss(self, model, frames_set_t, frame_embeddings, batch_size, points_per_pair, fg_points_ratio=0.5, temp=0.5, cl_div=800):
        source_selector = torch.randint(frames_set_t.shape[0], (batch_size,), device=frames_set_t.device) # batch_size
        target_selector = torch.randint(frames_set_t.shape[0], (batch_size,), device=frames_set_t.device) # batch_size
        feat_coord_grid = get_vit_feature_coords_from_mask(h=model.video.shape[-2], w=model.video.shape[-1],
                                                           step=model.stride, patch_size=self.config["dino_patch_size"]).to(frame_embeddings.device) # n x 2
        
        BATCH_SIZE_PER_FRAME = points_per_pair
        BATCH_SIZE_FG = int(BATCH_SIZE_PER_FRAME * fg_points_ratio)
        BATCH_SIZE_BG = BATCH_SIZE_PER_FRAME - BATCH_SIZE_FG
        
        n_total_bb = 0
        loss_contrastive_total = 0
        for source_idx, target_idx in zip(source_selector, target_selector):
            source_f = rearrange(frame_embeddings[source_idx], 'c h w -> (h w) c') # n x c, n = hxw
            target_f = rearrange(frame_embeddings[target_idx], 'c h w -> (h w) c') # n x c, n = hxw
            
            feature_range = torch.arange(source_f.shape[0]).to(source_f.device)

            with torch.no_grad():
                # affinity = torch.einsum("nc,mc->nm", source_f, target_f)
                MAX_AFFINITY_BATCH_SIZE  = 10_000
                if source_f.shape[0] > MAX_AFFINITY_BATCH_SIZE:
                    print('affinity batched')
                    affinity = torch.zeros(source_f.shape[0], target_f.shape[0], device=source_f.device)
                    affinity_batch_size = MAX_AFFINITY_BATCH_SIZE # 10_000
                    for i in range(0, source_f.shape[0], affinity_batch_size):
                        end_idx = min(i+affinity_batch_size, source_f.shape[0])
                        source_f_batch = source_f[i:end_idx]
                        affinity[i:end_idx] = torch.einsum("nc,mc->nm", source_f_batch, target_f)
                        affinity[i:end_idx] = affinity[i:end_idx] / torch.clamp(source_f_batch.norm(dim=1)[:, None] * target_f.norm(dim=1)[None, ...], min=1e-08)  
                else:
                    affinity = torch.einsum("nc,mc->nm", source_f, target_f)
                    affinity = affinity / torch.clamp(source_f.norm(dim=1)[:, None] * target_f.norm(dim=1)[None, ...], min=1e-08)
                
                affinity_source_max = torch.argmax(affinity, dim=1) # n
                affinity_target_max = torch.argmax(affinity, dim=0) # m
                source_bb_indices = feature_range == affinity_target_max[affinity_source_max] # bool of shape n, where b values are positive
                target_bb_indices = affinity_source_max[source_bb_indices] # b
                if source_bb_indices.sum() == 0:
                    continue
            
            # balance foreground-background
            source_bb_coords_current = feat_coord_grid[source_bb_indices]
            target_bb_coords_current = feat_coord_grid[target_bb_indices]
            _, _, fg_mask_source  = filter_bb_foreground_pairs(source_bb_coords_current,
                                                               target_bb_coords_current,
                                                               self.fg_masks[frames_set_t[source_idx]],
                                                               resw=model.video.shape[-1],
                                                               resh=model.video.shape[-2])
            source_bb_f_fg = source_f[source_bb_indices][fg_mask_source]
            source_bb_f_bg = source_f[source_bb_indices][~fg_mask_source]
            target_bb_f_fg = target_f[target_bb_indices][fg_mask_source]
            target_bb_f_bg = target_f[target_bb_indices][~fg_mask_source]
            
            # select random pairs
            fg_selector = torch.randperm(source_bb_f_fg.shape[0])[:BATCH_SIZE_FG]
            bg_selector = torch.randperm(target_bb_f_bg.shape[0])[:BATCH_SIZE_BG]
            source_bb_f_fg_sampled = source_bb_f_fg[fg_selector]
            target_bb_f_fg_sampled = target_bb_f_fg[fg_selector]
            source_bb_f_bg_sampled = source_bb_f_bg[bg_selector]
            target_bb_f_bg_sampled = target_bb_f_bg[bg_selector]
            
            source_bb_indices_num = torch.arange(source_bb_indices.shape[0], device=source_bb_indices.device)[source_bb_indices]
            source_selector = torch.cat([source_bb_indices_num[fg_mask_source][fg_selector],
                                         source_bb_indices_num[~fg_mask_source][bg_selector]])
            target_selector = torch.cat([target_bb_indices[fg_mask_source][fg_selector], target_bb_indices[~fg_mask_source][bg_selector]])
            
            cl1, cl2, _, _ = self.get_bb_pairs_contrastive_loss(
                source_bb_f=torch.cat([source_bb_f_fg_sampled, source_bb_f_bg_sampled]),
                target_bb_f=torch.cat([target_bb_f_fg_sampled, target_bb_f_bg_sampled]),
                source_f=source_f,
                target_f=target_f,
                temp=temp
            )
            with torch.no_grad():
                w_cos_sim = torch.clamp(2*(affinity[source_selector, target_selector]**3), 0) # cosine weighting
                w_cos_sim = w_cos_sim.detach()
            n_total_bb += 2 * cl1.shape[0]
            loss_contrastive_total += (cl1 * w_cos_sim).sum() + (cl2 * w_cos_sim).sum()
        
        if n_total_bb == 0:
            return torch.tensor(0.).to(frame_embeddings.device)
        
        cl_divisor = 2*cl_div
        loss_contrastive_total /= cl_divisor
        return loss_contrastive_total

    def get_bb_pairs_contrastive_loss(self, source_bb_f, target_bb_f, source_f, target_f, temp=0.5):
        bb_corrs = torch.einsum('bc,bc->b', source_bb_f, target_bb_f) # b
        source_target_corrs = torch.einsum('bc,nc->bn', source_bb_f, target_f)
        target_source_corrs = torch.einsum('bc,nc->bn', target_bb_f, source_f)
        source_target_corrs = source_target_corrs / torch.clamp(source_bb_f.norm(dim=1)[:, None] * target_f.norm(dim=1)[None, ...],
                                                                min=1e-08) # b x n
        target_source_corrs = target_source_corrs / torch.clamp(target_bb_f.norm(dim=1)[:, None] * source_f.norm(dim=1)[None, ...],
                                                                min=1e-08) # b x n
        bb_corrs /= torch.clamp(source_bb_f.norm(dim=1) * target_bb_f.norm(dim=1), min=1e-08)
        loss_source_target = -torch.log(torch.exp(bb_corrs / temp) / torch.exp(source_target_corrs / temp).sum(dim=1))
        loss_target_source = -torch.log(torch.exp(bb_corrs / temp) / torch.exp(target_source_corrs / temp).sum(dim=1))
        
        return loss_source_target, loss_target_source, bb_corrs.mean(), (source_target_corrs.mean() + target_source_corrs.mean()) / 2
    
    def get_cycle_consistency_loss(self, model, inputs):
        cycle_consistency_preds = model.get_cycle_consistent_preds(inputs[-1], self.fg_masks)
        consistent_track_weight = self.config["cyc_gamma"] ** cycle_consistency_preds["cycle_consistency_dists"]
        source_target_tracking_loss = consistent_track_weight[:, None] * self.of_loss_fn(cycle_consistency_preds["source_target_coords"], cycle_consistency_preds["target_coords"][:, :2])
        target_source_tracking_loss = consistent_track_weight[:, None] * self.of_loss_fn(cycle_consistency_preds["target_source_coords"], cycle_consistency_preds["source_coords"][:, :2])
        consistent_track_loss = (source_target_tracking_loss.mean() + target_source_tracking_loss.mean()) / 2
        
        return consistent_track_loss
    
    def init_losses(self):
        self.running_loss_total = 0.
        self.running_loss_of = 0.
        self.running_loss_cl_dino_bb = 0.
        self.running_loss_cl_refiner = 0.
        self.running_loss_emb_norm_reg = 0.
        self.running_loss_angle_reg = 0.
        self.running_loss_cyc = 0.
        
    def update_losses(self, loss_total, loss_of, loss_cl_dino_bb, loss_cl_refiner, loss_emb_norm_reg, loss_angle_reg, loss_cyc):
        self.running_loss_total += loss_total
        self.running_loss_of += loss_of
        self.running_loss_cl_dino_bb += loss_cl_dino_bb
        self.running_loss_cl_refiner += loss_cl_refiner
        self.running_loss_emb_norm_reg += loss_emb_norm_reg
        self.running_loss_angle_reg += loss_angle_reg
        self.running_loss_cyc += loss_cyc
    
    def log_losses(self, i, log_interval=100):
        loss_of = self.running_loss_of / log_interval
        loss_cl_dino_bb = self.running_loss_cl_dino_bb / log_interval
        loss_emb_norm_reg = self.running_loss_emb_norm_reg / log_interval
        loss_angle_reg = self.running_loss_angle_reg / log_interval
        loss_cl_refiner = self.running_loss_cl_refiner / log_interval if i >= self.config.get("apply_cl_ref_after", 0) else None
        loss_cyc = self.running_loss_cyc / log_interval if i >= self.config.get("apply_cyc_after", 0) else None
        loss_total = self.running_loss_total / log_interval
        
        loss_str = f"loss_of: {loss_of:.4f}, loss_cl_dino_bb: {loss_cl_dino_bb:.4f}, loss_emb_norm_reg: {loss_emb_norm_reg:.4f}, loss_angle_reg: {loss_angle_reg:.4f}"
        if loss_cl_refiner is not None:
            loss_str += f", loss_cl_refiner: {loss_cl_refiner:.4f}"
        if loss_cyc is not None:
            loss_str += f", loss_cyc: {loss_cyc:.4f}"
        loss_str += f", loss_total: {loss_total:.4f}"
        
        logging.info(loss_str)
        self.init_losses()

    def train(self):
        self.load_fg_masks()
        # Get values from config
        total_iterations = self.config["total_iterations"]
        checkpoint_interval = self.config["checkpoint_interval"]
        sampler_batch_iterations = self.config.get("sampler_batch_iterations", 100_000) # only relevant if in config, 100k is never reached

        self.load_dino_best_buddies()
        train_sampler = self.get_sampler()
        model, optimizer, scheduler = self.train_setup()
        self.set_model_train(model)
        self.init_losses()
        
        for i in tqdm(range(self.init_iter, total_iterations)):
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            inputs, labels = self.get_inputs_and_labels(train_sampler)
            
            coords = model(inputs)
            tracking_loss = self.of_loss_fn(coords, labels).mean()
            loss = tracking_loss
            
            if i >= self.config.get("apply_cyc_after", 0):
                consistent_track_loss = self.get_cycle_consistency_loss(model, inputs)
                loss += self.config["lambda_cyc"] * consistent_track_loss

            if i >= self.config.get("apply_cl_ref_after", 0):
                loss_cl_refiner = self.get_refiner_contrastive_loss(model, frames_set_t=inputs[-1])
                loss += self.config["lambda_cl_ref_bb"] * loss_cl_refiner

            loss_cl_dino_bb = self.get_dino_bb_contrastive_loss(model, frames_set_t=inputs[-1])
            loss_emb_norm_reg = self.get_emb_norm_regularization_loss(model.module if hasattr(model, "module") else model)
            loss_angle_reg = self.get_emb_angle_regularization_loss(model.module if hasattr(model, "module") else model)
            
            loss += self.config["lambda_cl_dino_bb"] * loss_cl_dino_bb + self.config["lambda_emb_norm"] * loss_emb_norm_reg + self.config["lambda_angle"] * loss_angle_reg
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # logging losses
            self.update_losses(loss.item(), tracking_loss.item(), loss_cl_dino_bb.item(),
                               loss_cl_refiner.item() if i >= self.config.get("apply_cl_ref_after", 0) else 0,
                               loss_emb_norm_reg.item(), loss_angle_reg.item(),
                               consistent_track_loss.item() if i >= self.config.get("apply_cyc_after", 0) else 0)
            if i % 100 == 0:
                self.log_losses(i, log_interval=100)
            
            # saving checkpoint
            if (i == total_iterations - 1 or i % checkpoint_interval == 0):
                model.save_weights(i)

            # loading next batch of trajectories
            if (i % sampler_batch_iterations == 0 and i > 0):
                print("Loading next batch", flush=True)
                train_sampler.load_next_batch()

        model.save_weights(total_iterations)
