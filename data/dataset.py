import math
import torch


class RangeNormalizer(torch.nn.Module):
    """
    Scales dimensions to specific ranges.
    Will be used to normalize pixel coords. & time to destination ranges.
    For example: [0, H-1] x [0, W-1] x [0, T-1] -> [0,1] x [0,1] x [0,1]

    Args:
         shapes (tuple): represents the "boundaries"/maximal values for each input dimension.
            We assume that the dimensions range from 0 to max_value (as in pixels & frames).
    """
    def __init__(self, shapes: tuple, device='cuda'):
        super().__init__()

        normalizer = torch.tensor(shapes).float().to(device) - 1
        self.register_buffer("normalizer", normalizer)

    def forward(self, x, dst=(0, 1), dims=[0, 1, 2]):
        """
        Normalizes input to specific ranges.
        
            Args:       
                x (torch.tensor): input data
                dst (tuple, optional): range inputs where normalized to. Defaults to (0, 1).
                dims (list, optional): dimensions to normalize. Defaults to [0, 1, 2].
                
            Returns:
                normalized_x (torch.tensor): normalized input data
        """
        normalized_x = x.clone()
        normalized_x[:, dims] = x[:, dims] / self.normalizer[dims] # normalize to [0,1]
        normalized_x[:, dims] = (dst[1] - dst[0]) * normalized_x[:, dims] + dst[0] # shift range to dst

        return normalized_x
    
    def unnormalize(self, normalized_x:torch.tensor, src=(0, 1), dims=[0, 1, 2]):
        """Runs to reverse process of forward, unnormalizes input to original scale.

        Args:
            normalized_x (torch.tensor): input data
            src (tuple, optional): range inputs where normalized to. Defaults to (0, 1). unnormalizes from src to original scales.
            dims (list, optional): dimensions to normalize. Defaults to [0, 1, 2].

        Returns:
            x (torch.tensor): unnormalized input data
        """
        x = normalized_x.clone()
        x[:, dims] = (normalized_x[:, dims] - src[0]) / (src[1] - src[0]) # shift range to [0,1]
        x[:, dims] = x[:, dims] * self.normalizer[dims] # unnormalize to original ranges
        return x


class LongRangeSampler(torch.nn.Module):
    def __init__(self,
                 batch_size,
                 fg_trajectories=None,
                 bg_trajectories=None,
                 fg_traj_ratio=0.5,
                 num_frames=None,
                 keep_in_cpu=False) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.num_frames = num_frames
        self.fg_traj_ratio = fg_traj_ratio
        
        # allow storing a large number of trajectories in cpu, and keep small batches in gpu for faster sampling
        self.keep_in_cpu = keep_in_cpu
        self.max_traj_size = 200_000
        self.gpu_batch_index = 0
        self.iter = 0
        
        # if trajectories not in cpu, keep in gpu and use it directly
        if not self.keep_in_cpu:
            self.fg_valid_trajectories, self.fg_can_sample = self.get_valid_trajectories(fg_trajectories)
            self.bg_valid_trajectories, self.bg_can_sample = self.get_valid_trajectories(bg_trajectories)
            self.vid_len = self.fg_valid_trajectories.shape[1]

        # if trajectories in cpu, keep in cpu and sample in batches from it
        else:
            self.fg_valid_trajectories_complete, self.fg_can_sample_complete = self.get_valid_trajectories(fg_trajectories)
            del fg_trajectories # free memory
            torch.cuda.empty_cache()
            # keep fg_valid_trajectories_complete in cpu and sample in batches from it
            self.fg_valid_trajectories = self.fg_valid_trajectories_complete[:self.max_traj_size].cuda()
            self.fg_can_sample = self.fg_can_sample_complete[:self.max_traj_size].cuda()
            self.n_batches_fg = math.ceil(self.fg_valid_trajectories_complete.shape[0] / self.max_traj_size)

            self.bg_valid_trajectories_complete, self.bg_can_sample_complete = self.get_valid_trajectories(bg_trajectories)
            del bg_trajectories # free memory
            torch.cuda.empty_cache()
            # keep bg_valid_trajectories_complete in cpu and sample in batches from it
            self.bg_valid_trajectories = self.bg_valid_trajectories_complete[:self.max_traj_size].cuda()
            self.bg_can_sample = self.bg_can_sample_complete[:self.max_traj_size].cuda()
            self.n_batches_bg = math.ceil(self.bg_valid_trajectories_complete.shape[0] / self.max_traj_size)
    
    def get_valid_trajectories(self, trajectories):
        can_sample = trajectories.isnan().any(dim=-1).logical_not() # N x t
        valid_trajs_idx = (can_sample.sum(dim=1) > 1) # N
        # Remove trajectories that are only one frame long
        valid_trajectories = trajectories[valid_trajs_idx] # N'xtx2
        can_sample = can_sample[valid_trajs_idx] # N'xt
        return valid_trajectories, can_sample

    def load_next_batch(self):
        if not self.keep_in_cpu:
            return
        
        self.gpu_batch_index += 1
        # check if self has an attribute called valid_trajectories_complete
        if hasattr(self, "valid_trajectories_complete"):
            del self.can_sample, self.valid_trajectories # free memory
            batch_index = self.gpu_batch_index % self.n_batches
            start_index, end_index = batch_index * self.max_traj_size, min((batch_index + 1) * self.max_traj_size, self.valid_trajectories_complete.shape[0])
            self.valid_trajectories = self.valid_trajectories_complete[start_index:end_index].cuda()
            self.can_sample = self.can_sample_complete[start_index:end_index].cuda()
        if hasattr(self, "fg_valid_trajectories_complete"):
            del self.fg_can_sample, self.fg_valid_trajectories
            fg_batch_index = self.gpu_batch_index % self.n_batches_fg
            start_index, end_index = fg_batch_index * self.max_traj_size, min((fg_batch_index + 1) * self.max_traj_size, self.fg_valid_trajectories_complete.shape[0])
            self.fg_valid_trajectories = self.fg_valid_trajectories_complete[start_index:end_index].cuda()
            self.fg_can_sample = self.fg_can_sample_complete[start_index:end_index].cuda()
        if hasattr(self, "bg_valid_trajectories_complete"):
            del self.bg_can_sample, self.bg_valid_trajectories
            bg_batch_index = self.gpu_batch_index % self.n_batches_bg
            start_index, end_index = bg_batch_index * self.max_traj_size, min((bg_batch_index + 1) * self.max_traj_size, self.bg_valid_trajectories_complete.shape[0])
            self.bg_valid_trajectories = self.bg_valid_trajectories_complete[start_index:end_index].cuda()
            self.bg_can_sample = self.bg_can_sample_complete[start_index:end_index].cuda()

    # used for legacy purposes
    @staticmethod
    def generate_start_end(trajectories):
        """Generates a start and end point for each trajectory

        Args:
            trajectories (torch.Tensor): A tensor of size Nxtx2 that contains point trajectory where optical flow was consistent and "nan" where it wasn't
        """
        mask = trajectories.isnan().any(dim=-1)

        # Generate start points
        mask_shifted_right = mask.roll(1, dims=1)
        mask_shifted_right[:, 0] = True

        first_timestep_mask = ~mask & mask_shifted_right
        first_timestep = first_timestep_mask.nonzero()[:, 1]

        # Generate end points
        mask_shifted_left = mask.roll(-1, dims=1)
        mask_shifted_left[:, -1] = True

        last_timestep_mask = ~mask & mask_shifted_left
        last_timestep = last_timestep_mask.nonzero()[:, 1]

        # Combine
        start_end = torch.stack([first_timestep, last_timestep], dim=1)

        return start_end
    
    def get_point_correspondences_for_num_frames(self, valid_trajectories, can_sample, batch_size):
        b, t, _ = valid_trajectories.shape
        
        done_selecting_frames = False
        
        while not done_selecting_frames:
            times = torch.arange(t, device=valid_trajectories.device)
            t_selector = torch.randperm(times.shape[0], device=valid_trajectories.device)[:self.num_frames]
            frame_indices = times[t_selector]
            can_sample_at_frame_indices = can_sample.float()[:, frame_indices].sum(dim=1) >= 2
            can_sample_current = can_sample[can_sample_at_frame_indices]
            if len(can_sample_current) >= 2:
                trajectories = valid_trajectories[can_sample_at_frame_indices]
                done_selecting_frames = True
        
        batch_size_selector = torch.randperm(trajectories.shape[0], device=valid_trajectories.device)[:batch_size]
        can_sample_current = can_sample_current[batch_size_selector]
        
        can_sample_in_frame_indices = can_sample_current[:, frame_indices]
        can_sample_current[:, :] = False
        can_sample_current[:, frame_indices] = can_sample_in_frame_indices
        only_2_ts = can_sample_current.float().multinomial(2, replacement=False) # batch_size x 2
        t1, t2 = only_2_ts.unbind(dim=1) # batch_size x 1
        t1_points = trajectories[batch_size_selector, t1]
        t2_points = trajectories[batch_size_selector, t2]
        t1_points = torch.cat([t1_points, t1.unsqueeze(dim=-1)], dim=-1)
        t2_points = torch.cat([t2_points, t2.unsqueeze(dim=-1)], dim=-1)
        
        return t1_points, t2_points
    
    def get_fg_batch_size(self):
        return int(self.batch_size * self.fg_traj_ratio)

    def forward(self):
        assert self.num_frames is not None, "num_frames must be specified"
        
        fg_batch_size = self.get_fg_batch_size()
        bg_batch_size = self.batch_size - fg_batch_size
        fg_t1_points, fg_t2_points = self.get_point_correspondences_for_num_frames(self.fg_valid_trajectories,
                                                                                    self.fg_can_sample,
                                                                                    fg_batch_size)
        bg_t1_points, bg_t2_points = self.get_point_correspondences_for_num_frames(self.bg_valid_trajectories,
                                                                                    self.bg_can_sample,
                                                                                    bg_batch_size)
        t1_points = torch.cat([fg_t1_points, bg_t1_points], dim=0)
        t2_points = torch.cat([fg_t2_points, bg_t2_points], dim=0)
        return t1_points, t2_points


class DinoTrackerSampler(LongRangeSampler):
    def __init__(
        self,
        batch_size,
        range_normalizer,
        dst_range,
        fg_trajectories=None,
        bg_trajectories=None,
        fg_traj_ratio=0.5,
        num_frames=None,
        keep_in_cpu=False,
        ) -> None:
        super().__init__(batch_size,
                         fg_trajectories=fg_trajectories,
                         bg_trajectories=bg_trajectories,
                         fg_traj_ratio=fg_traj_ratio,
                         num_frames=num_frames,
                         keep_in_cpu=keep_in_cpu)

        self.range_normalizer = range_normalizer
        self.dst_range = dst_range

    def forward(self):

        # Sample t1 & t2 points using LongRangeSampler
        t1_points, t2_points = super().forward()

        # make a set out of t1_points
        frames_set_t = torch.cat((t1_points[:, 2], t2_points[:, 2])).unique().int() # shape: T
        source_frame_indices = torch.cat( [ (frames_set_t==i).nonzero() for i in t1_points[:, 2] ] )[:, 0] # shape: B
        target_frame_indices = torch.cat( [ (frames_set_t==i).nonzero() for i in t2_points[:, 2] ] )[:, 0] # shape: B

        # normalize (x,y,t) ranges to [-1,1]
        t1_points_normalized = self.range_normalizer(t1_points, dst=self.dst_range)
        t2_points_normalized = self.range_normalizer(t2_points, dst=self.dst_range)
        t1_points[:, 2] = t1_points_normalized[:, 2]
        
        sample = {
            "frames_set_t": frames_set_t, # shape: T
            "source_frame_indices": source_frame_indices, # shape: B
            "target_frame_indices": target_frame_indices, # shape: B
            "t1_points_normalized": t1_points_normalized, # shape: B x 3
            "t2_points_normalized": t2_points_normalized, # shape: B x 3
            "t1_points": t1_points, # shape: B x 3
            "target_times": t2_points[:, 2] # shape: B
        }
        
        return sample
