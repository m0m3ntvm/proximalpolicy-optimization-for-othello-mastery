"""Video writers: MP4 via imageio-ffmpeg, animated GIF via Pillow."""

import numpy as np


def _as_uint8(frames):
    return [np.asarray(f, dtype=np.uint8) for f in frames]


def write_gif(frames, path, fps=2):
    """Write uint8 RGB frames as a looping animated GIF (works without ffmpeg)."""
    from PIL import Image
    images = [Image.fromarray(f) for f in _as_uint8(frames)]
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=int(round(1000 / fps)), loop=0)
    return path


def write_video(frames, path, fps=2):
    """Write uint8 RGB frames to `path`.

    '.gif' always works (Pillow); other extensions (e.g. '.mp4') require
    imageio + imageio-ffmpeg (`uv add 'imageio[ffmpeg]'`).
    """
    if path.lower().endswith('.gif'):
        return write_gif(frames, path, fps)
    import imageio.v2 as imageio  # ImportError -> clear message, install imageio[ffmpeg]
    imageio.mimsave(path, _as_uint8(frames), fps=fps)
    return path
