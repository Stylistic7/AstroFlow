import os
import numpy as np
import cv2
from functools import wraps
from typing import Callable, Optional
import tempfile

from hashing import hash_call, hash_source_code
from protocols import CacheProtocol


def make_create_video_decorator(cache: CacheProtocol):
    def create_video(cache: CacheProtocol):
        def decorator(
            func: Callable, fourcc_str: str = "VP80", frame_rate: float = 10.0
        ):
            fn_hash = hash_source_code(func)

            @wraps(func)
            def wrapper(*args, **kwargs):
                frame_generator = func(*args, **kwargs)

                with tempfile.TemporaryDirectory() as tmp_dir:
                    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)  # type: ignore
                    video_filename = os.path.join(tmp_dir, "output.webm")

                    frame = next(frame_generator)
                    height, width = frame.shape
                    height = int(((height + 8 - 1) / 8) * 8)

                    def pad(im: np.ndarray):
                        frame = np.zeros((height, width), dtype=np.uint8)
                        frame[0 : im.shape[0], 0 : im.shape[1]] = im
                        return frame

                    out = cv2.VideoWriter(
                        video_filename, fourcc, frame_rate, (width, height), False
                    )
                    out.write(pad(frame))

                    for frame in frame_generator:
                        out.write(pad(frame))

                    out.release()
                    video = open(video_filename, "rb").read()

                    call_hash = hash_call(fn_hash, *args, **kwargs)
                    video_key = ("video", call_hash)
                    cache[video_key] = video
                    print(f"### create_video wrote video key {video_key}")

            def get_video_fn(*args, **kwargs) -> Optional[np.ndarray]:
                call_hash = hash_call(fn_hash, *args, **kwargs)
                video_key = ("video", call_hash)
                return cache.get(video_key)

            wrapper.video_fn = get_video_fn
            wrapper.__annotations__['return'] = type(None)

            return wrapper

        return decorator

    return create_video(cache)
