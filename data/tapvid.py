import pickle
from typing import Union


def get_video_config_by_video_id(benchmark_config: dict, video_id:int) -> dict:
    """Get video config by video id
    Args:
        benchmark_config (dict): benchmark config dictionary, in TAPVid format.
        video_id (int): video id
    Returns:
        dict: video config.
    """
    for video_config in benchmark_config["videos"]:
        if video_config["video_idx"] == video_id:
            return video_config
    return None


def get_query_points_from_benchmark_config(benchmark_config:Union[str, dict], video_idx:int, rescale_sizes=None) -> dict:
    """Get query points from config file
    Args:
        benchmark_config (Union[str, dict]): benchmark config dict (or path to config), for all vidoes. adds frame_idx to the points. (x, y) to (x, y, frame_idx).
        query points in tapvid format:
            {frame_idx: [[x1,y1], [x2,y2], ...[xn,yn]], size=(N,3)}.
    Returns:
        dict: query points {frame_idx: [[x1,y1,frame_idx], [x2,y2,frame_idx], ...[xn,yn,frame_idx]], size=(N,3)}.
    """
    benchmark_config = pickle.load(open(benchmark_config, "rb")) if type(benchmark_config) == str else benchmark_config
    video_config = get_video_config_by_video_id(benchmark_config, video_idx)
    rescale_factor_x = 1 if rescale_sizes is None else (rescale_sizes[0] / video_config['w'])
    rescale_factor_y = 1 if rescale_sizes is None else (rescale_sizes[1] / video_config['h'])
    
    query_points_dict = {}
    
    for frame_idx, q_pts_at_frame in video_config['query_points'].items():
        query_points_at_frame = []
        for q_point in q_pts_at_frame:
            query_points_at_frame.append([rescale_factor_x * q_point[0], rescale_factor_y * q_point[1], frame_idx])
        query_points_dict[frame_idx] = query_points_at_frame

    return query_points_dict
