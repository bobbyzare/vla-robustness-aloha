"""Turn an evaluation video (MP4) into a small looping GIF for the README.

GitHub plays GIFs inline in a README, but not MP4 files. Evaluation videos are
640x480 at 50 fps; for the README we shrink them so the page loads quickly and
the repo stays light.

Usage:
    conda activate lerobot && unset PYTHONPATH
    python scripts/make_gif.py <input.mp4> <output.gif>
    python scripts/make_gif.py in.mp4 media/out.gif --scale 2 --frame-step 2

Defaults: half size (320x240) and every 2nd frame (25 fps), so the GIF still
plays at real speed.
"""

import argparse
from pathlib import Path

import imageio.v3 as iio

VIDEO_FPS = 50  # gym-aloha / lerobot-eval videos: one frame per 0.02 s sim step


def make_gif(src, dst, scale=2, frame_step=2):
    """Read src (MP4), downscale and subsample it, and write a looping GIF to dst."""
    frames = iio.imread(src)  # all frames, shape (T, H, W, 3), uint8

    # Keep every `frame_step`-th frame and every `scale`-th pixel row/column.
    # Plain pixel skipping is enough here: the scene is flat-shaded, so no
    # visible aliasing, and it needs no extra image library.
    small = frames[::frame_step, ::scale, ::scale]

    # Show each GIF frame for as long as the frames it replaces, so playback
    # stays at real speed (frame_step=2 at 50 fps -> 40 ms per GIF frame).
    duration_ms = 1000 * frame_step / VIDEO_FPS

    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(dst, small, duration=duration_ms, loop=0)  # loop=0: repeat forever
    return small.shape, Path(dst).stat().st_size


def main():
    parser = argparse.ArgumentParser(description="Convert an evaluation MP4 into a small README GIF.")
    parser.add_argument("src", help="input video, e.g. outputs/eval/.../eval_episode_4.mp4")
    parser.add_argument("dst", help="output GIF, e.g. media/act_success.gif")
    parser.add_argument("--scale", type=int, default=2, help="keep every n-th pixel (2 = half size)")
    parser.add_argument("--frame-step", type=int, default=2, help="keep every n-th frame (2 = 25 fps)")
    args = parser.parse_args()

    shape, size = make_gif(args.src, args.dst, args.scale, args.frame_step)
    print(f"{args.dst}: {shape[0]} frames of {shape[2]}x{shape[1]}, {size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
