import argparse
import os
import torch
from torchvision.ops import batched_nms #, nms
from torch import einsum
from tqdm import tqdm
from preprocessing_dino_bb.dino_bb_utils import create_meshgrid, xy_to_fxy

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_bb_sim_indices(affs_batched, coords, box_size=50, iou_thresh=0.5, topk=400, device=device):
    """  affs_batched: B x N """
    topk = torch.topk(affs_batched, k=topk, sorted=False, dim=1)
    filt_idx = topk.indices # B x topk
    affs_filt = topk.values # B x topk

    if affs_filt.shape[0] == 0:
        return None, None, None
    
    filt_coords = coords[filt_idx]
    xmin = filt_coords[:, :, 0] - box_size # B x topk
    xmax = filt_coords[:, :, 0] + box_size # B x topk
    ymin = filt_coords[:, :, 1] - box_size # B x topk
    ymax = filt_coords[:, :, 1] + box_size # B x topk
    # concat to get boxes shaped B x topk x 4
    boxes = torch.cat([xmin[:, :, None], ymin[:, :, None], xmax[:, :, None], ymax[:, :, None]], dim=-1) # B x topk x 4
    # get idxs shaped (B x topk) representing the batch index
    idxs = torch.arange(filt_idx.shape[0], device=device)[:, None].repeat(1, filt_idx.shape[1]).reshape(-1) # (B x topk)
    peak_indices = batched_nms(boxes.reshape(-1, 4), affs_filt.reshape(-1), idxs, iou_thresh)
    # convert peak_indices to the original indices to the  not flat indices
    peak_indices_original = torch.stack([peak_indices // filt_idx.shape[1], peak_indices % filt_idx.shape[1]], dim=-1)
    # retrieve the first two elements of the peak_indices_original for the first axis
    filt_idx_mask = torch.zeros_like(filt_idx, device=device) # B x topk
    filt_idx_mask[peak_indices_original[:, 0], peak_indices_original[:, 1]] = 1
    peak_aff_batched = affs_filt * filt_idx_mask # B x topk
    # retrieve the highest and second highest affinities for each batch
    top2 = torch.topk(peak_aff_batched, k=2, dim=1)
    top2_values, top2_indices = top2.values, top2.indices # B x 2, B x 2
    highest_affs, highest_affs_idx = top2_values[:, 0], top2_indices[:, 0] # B, B
    second_highest_affs, second_highest_affs_idx = top2_values[:, 1], top2_indices[:, 1] # B, B
    r = second_highest_affs / highest_affs 
    return None, top2_values, r


def compute_bb_nms(dino_bb_sf_tf, sf, tf, dino_emb, coords, stride, box_size, iou_thresh):
    source_xy = dino_bb_sf_tf['source_coords']
    source_fxy = xy_to_fxy(source_xy, stride=stride) # N x 2

    target_fmap = dino_emb[tf] # C x H x W
    source_f = dino_emb[sf, :, source_fxy[:, 1].int(), source_fxy[:, 0].int()] # C x N
    
    source_target_sim = einsum('cn,chw->nhw', source_f, target_fmap)
    source_f_norm = torch.norm(source_f, dim=0) # N
    target_fmap_norm = torch.norm(target_fmap, dim=0) # H x W
    source_target_sim /= torch.clamp(source_f_norm[:, None, None] * target_fmap_norm[None, :, :], min=1e-08) # N x H x W
    N = source_target_sim.shape[0]

    source_target_sim_flat = source_target_sim.reshape(N, -1) # N x (HxW)
    
    _, peak_aff, r = get_bb_sim_indices(source_target_sim_flat, coords, box_size=box_size, iou_thresh=iou_thresh)
    dino_bb_sf_tf['peak_coords'] = None
    dino_bb_sf_tf['peak_affs'] = peak_aff # N x 2
    dino_bb_sf_tf['r'] = r # N

    return dino_bb_sf_tf

def compute_max_r(bb, bb_rev):
    for i in range(bb['target_coords'].shape[0]):
        r = bb['r'][i]
        target_coord = bb['target_coords'][i]
        rev_idx = torch.norm(bb_rev['source_coords'] - target_coord[None, :], dim=1).argmin(0)
        assert torch.norm(bb_rev['target_coords'][rev_idx] - bb['source_coords'][i]) == 0
        rev_r = bb_rev['r'][rev_idx]
        max_r = max(rev_r, r)
        bb['r'][i] = max_r
        bb_rev['r'][rev_idx] = max_r
    return bb, bb_rev


def run(args):
    dino_bb = torch.load(args.dino_bb_path) # { 'i_j': { source_coords: [N x 2] } }
    dino_emb = torch.load(args.dino_emb_path) # t x c x h x w
    coords = create_meshgrid(h=476, w=854, step=args.stride) # N x 2

    for key in tqdm(dino_bb.keys()):
        # no dino-bbs for this frame pair
        if dino_bb[key]['source_coords'] is None:
            dino_bb[key]['peak_coords'] = None
            dino_bb[key]['peak_affs'] = None
            dino_bb[key]['r'] = None
            continue
        
        # nms already computed as bb_rev
        if dino_bb[key].get('r', None) is not None:
            continue

        sf, tf = int(key.split("_")[0]), int(key.split("_")[1])
        
        bb = compute_bb_nms(dino_bb[f'{sf}_{tf}'], sf, tf, dino_emb, coords, args.stride, args.box_size, args.iou_thresh)
        bb_rev = compute_bb_nms(dino_bb[f'{tf}_{sf}'], tf, sf, dino_emb, coords, args.stride, args.box_size, args.iou_thresh)
        
        bb, bb_rev = compute_max_r(bb, bb_rev)
        
        dino_bb[key] = bb
        dino_bb[f'{tf}_{sf}'] = bb_rev

    dino_bb_nms_path = args.out_path
    os.makedirs(os.path.dirname(dino_bb_nms_path), exist_ok=True)
    torch.save(dino_bb, dino_bb_nms_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dino-bb-path", type=str, required=True)
    parser.add_argument("--dino-emb-path", type=str, required=True)
    parser.add_argument("--out-path", type=str, required=True)
    parser.add_argument("--stride", type=int, default=7)
    parser.add_argument("--box-size", type=int, default=50)
    parser.add_argument("--iou-thresh", type=float, default=0.2)
    args = parser.parse_args()
    run(args)
