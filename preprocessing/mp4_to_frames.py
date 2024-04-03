import argparse
import os
import imageio


def mp4_to_frames(mp4_file, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    vid = imageio.get_reader(mp4_file)
    for i, frame in enumerate(vid):
        imageio.imwrite(os.path.join(output_folder, f"{i:05d}.jpg"), frame)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", type=str)
    parser.add_argument("--output-folder", type=str)
    args = parser.parse_args()
    mp4_to_frames(args.video_path, args.output_folder)

# python mp4_to_frames.py --video-path ./dataset/libby-test/visualizations/dotted_tracks_fps_10.mp4 --output-folder /dataset/libby-test/visualizations-frames