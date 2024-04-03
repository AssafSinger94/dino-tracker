import os 
from preprocessing_dino_bb.dino_bb_utils import create_meshgrid
import torch
import argparse
from tqdm import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_closest_traj_idx_batch(trajectories, points, t, batch_size=100):
    """ 
    Args:
        trajectories: N x T x 2
        points: B x 2
        t: int
    returns: B
        """
    # sample trajectories at time t
    trajectories_at_t = trajectories[:, t, :] # N x 2
    # iterate over points in batches of size batch_size
    closest_traj_idx_list = []
    for i in range(0, len(points), batch_size):
        end = min(i + batch_size, len(points))
        points_batch = points[i:end]
        source_dist = torch.norm(trajectories_at_t[None, ...] - points_batch[:, None, :], dim=2) # B x N
        source_dist = torch.nan_to_num(source_dist, nan=torch.inf)
        # comput argmin on the last dimension
        closest_traj_idx_list.append(source_dist.argmin(dim=-1)) # B
    closest_traj_idx = torch.cat(closest_traj_idx_list, dim=0)
    return closest_traj_idx


def is_point_valid(point):
    return not point.isnan().any()


@torch.no_grad()
def run(args):
    
    dino_bb_path = args.dino_bb_path
    traj_path = args.traj_path
    out_path = args.out_path
    dino_bb_stride = args.dino_bb_stride
    h, w = args.h, args.w
    
    bb_data = torch.load(dino_bb_path)
    traj = torch.load(traj_path).to(device)

    # pre-compute closest traj indices for each point
    video_len = traj.shape[1]
    # grid, w_grid, h_grid = create_meshgrid(h, w, step=dino_bb_stride, return_hw=True)
    grid, h_grid, w_grid = create_meshgrid(h, w, step=dino_bb_stride, return_hw=True)
    closest_traj_idx_grid_dict = {}
    for t in tqdm(range(video_len), desc="pre-computing trajectory indices"):
        closest_traj_idx = get_closest_traj_idx_batch(traj, grid, t, 30) # B, that is, len(grid)
        closest_traj_idx_grid = closest_traj_idx.reshape(h_grid, w_grid) # H' x W'
        closest_traj_idx_grid_dict[t] = closest_traj_idx_grid
    # free up memory
    traj = traj.cpu()
    torch.cuda.empty_cache()

    total_filtered_bb = {}
    traj_is_point_invalid = traj.isnan().any(dim=-1).to(device) # N x T
    del traj # free up memory

    for source_t in tqdm(range(video_len), desc="source frames"):
        for target_t in tqdm(range(video_len), desc="target frames"):
            if source_t == target_t:
                continue

            source_points = bb_data[f'{source_t}_{target_t}']['source_coords'] # N x 2; image coordinates
            target_points = bb_data[f'{source_t}_{target_t}']['target_coords'] # N x 2; image coordinates
            cos_sims = bb_data[f'{source_t}_{target_t}']['cos_sims']
            peak_coords = bb_data[f'{source_t}_{target_t}'].get('peak_coords', None)
            peak_affs = bb_data[f'{source_t}_{target_t}'].get('peak_affs', None)
            r = bb_data[f'{source_t}_{target_t}'].get('r', None)

            filtered_bb = {
                'source_coords': None,
                'target_coords': None,
                'cos_sims': None,
                'peak_coords' : None,
                'peak_affs' : None,
                 'r' : None,
            }

            # TODO: transform source_points to feature coordinates: (source_points - 7) / stride; repeat for target_pointss
            source_points_grid_idx = ((source_points - 7) // dino_bb_stride).long() # N x 2, in (x, y) format
            target_points_grid_idx = ((target_points - 7) // dino_bb_stride).long() # N x 2, in (x, y) format
            # replace x & y coordinates with closest traj indices
            # sample 
            closest_traj_idx_grid_at_source_t = closest_traj_idx_grid_dict[source_t] # H' x W'
            closest_traj_idx_grid_at_target_t = closest_traj_idx_grid_dict[target_t] # H' x W'
            # TODO: sample traj_indices_grid[source_points] --> source_traj_indices N x 1
            source_points_traj_indices = closest_traj_idx_grid_at_source_t[source_points_grid_idx[:, 1], source_points_grid_idx[:, 0]] # N
            target_points_traj_indices = closest_traj_idx_grid_at_target_t[target_points_grid_idx[:, 1], target_points_grid_idx[:, 0]] # N
            # TODO: traj_is_point_valid[source_traj_indices, target_t]
            should_sample_bb = (traj_is_point_invalid[source_points_traj_indices, target_t] & traj_is_point_invalid[target_points_traj_indices, source_t])
            # TODO: repeat the above for target_points
            filtered_bb['source_coords'] = source_points[should_sample_bb] if should_sample_bb.any() else None
            filtered_bb['target_coords'] = target_points[should_sample_bb] if should_sample_bb.any() else None
            filtered_bb['cos_sims'] = cos_sims[should_sample_bb] if should_sample_bb.any() else None
            if peak_coords is not None:
                filtered_bb['peak_coords'] = peak_coords[should_sample_bb] if should_sample_bb.any() else None
            if peak_affs is not None:
                filtered_bb['peak_affs'] = peak_affs[should_sample_bb] if should_sample_bb.any() else None
            if r is not None:
                filtered_bb['r'] = r[should_sample_bb] if should_sample_bb.any() else None
            total_filtered_bb[f'{source_t}_{target_t}'] = filtered_bb
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(total_filtered_bb, out_path)
    print(f"Saved filtered best buddies to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino-bb-path", type=str, required=True)
    parser.add_argument("--traj-path", type=str, required=True)
    parser.add_argument("--out-path", type=str, required=True)
    parser.add_argument("--dino-bb-stride", type=int, default=7)
    parser.add_argument("--h", type=int, default=476)
    parser.add_argument("--w", type=int, default=854)
    args = parser.parse_args()
    run(args)
