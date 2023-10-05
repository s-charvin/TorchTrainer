
import warnings
from typing import Optional, Union
from pathlib import Path
import io
import numpy as np
import os.path as osp

from .base import BaseTransform
from TorchTrainer.utils.registry import TRANSFORMS
import TorchTrainer.utils.fileio as fileio
from TorchTrainer.utils import is_filepath, is_str


import cv2
from cv2 import (
    IMREAD_COLOR,
    IMREAD_GRAYSCALE,
    IMREAD_IGNORE_ORIENTATION,
    IMREAD_UNCHANGED,
)


try:
    from turbojpeg import TJCS_RGB, TJPF_BGR, TJPF_GRAY, TurboJPEG
except ImportError:
    TJCS_RGB = TJPF_GRAY = TJPF_BGR = TurboJPEG = None


try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None

try:
    import tifffile
except ImportError:
    tifffile = None


import librosa

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

try:
    import av
except ImportError:
    av = None

try:
    import decord
except ImportError:
    decord = None

jpeg = None
im_supported_backends = ["cv2", "turbojpeg", "pillow", "tifffile"]

imread_flags = {
    "color": IMREAD_COLOR,
    "grayscale": IMREAD_GRAYSCALE,
    "unchanged": IMREAD_UNCHANGED,
    "color_ignore_orientation": IMREAD_IGNORE_ORIENTATION | IMREAD_COLOR,
    "grayscale_ignore_orientation": IMREAD_IGNORE_ORIENTATION | IMREAD_GRAYSCALE,
}

imread_backend = "cv2"


def _jpegflag(flag: str = "color", channel_order: str = "bgr"):
    channel_order = channel_order.lower()
    if channel_order not in ["rgb", "bgr"]:
        raise ValueError('channel order must be either "rgb" or "bgr"')

    if flag == "color":
        if channel_order == "bgr":
            return TJPF_BGR
        elif channel_order == "rgb":
            return TJCS_RGB
    elif flag == "grayscale":
        return TJPF_GRAY
    else:
        raise ValueError('flag must be "color" or "grayscale"')


def _pillow2array(img, flag: str = "color", channel_order: str = "bgr") -> np.ndarray:
    """Convert a pillow image to numpy array."""
    channel_order = channel_order.lower()
    if channel_order not in ["rgb", "bgr"]:
        raise ValueError('channel order must be either "rgb" or "bgr"')

    if flag == "unchanged":
        array = np.array(img)
        if array.ndim >= 3 and array.shape[2] >= 3:  # color image
            array[:, :, :3] = array[:, :, (2, 1, 0)]  # RGB to BGR
    else:
        # Handle exif orientation tag
        if flag in ["color", "grayscale"]:
            img = ImageOps.exif_transpose(img)
        # If the image mode is not 'RGB', convert it to 'RGB' first.
        if img.mode != "RGB":
            if img.mode != "LA":
                # Most formats except 'LA' can be directly converted to RGB
                img = img.convert("RGB")
            else:
                # When the mode is 'LA', the default conversion will fill in
                #  the canvas with black, which sometimes shadows black objects
                #  in the foreground.
                #
                # Therefore, a random color (124, 117, 104) is used for canvas
                img_rgba = img.convert("RGBA")
                img = Image.new("RGB", img_rgba.size, (124, 117, 104))
                img.paste(img_rgba, mask=img_rgba.split()[3])  # 3 is alpha
        if flag in ["color", "color_ignore_orientation"]:
            array = np.array(img)
            if channel_order != "rgb":
                array = array[:, :, ::-1]  # RGB to BGR
        elif flag in ["grayscale", "grayscale_ignore_orientation"]:
            img = img.convert("L")
            array = np.array(img)
        else:
            raise ValueError(
                'flag must be "color", "grayscale", "unchanged", '
                f'"color_ignore_orientation" or "grayscale_ignore_orientation"'
                f" but got {flag}"
            )
    return array


def imfrombytes(
    content: bytes,
    flag: str = "color",
    channel_order: str = "bgr",
    backend: Optional[str] = None,
) -> np.ndarray:
    """Read an image from bytes."""

    if backend is None:
        backend = imread_backend
    if backend not in im_supported_backends:
        raise ValueError(
            f"backend: {backend} is not supported. Supported "
            "backends are 'cv2', 'turbojpeg', 'pillow', 'tifffile'"
        )
    if backend == "turbojpeg":
        img = jpeg.decode(content, _jpegflag(flag, channel_order))  # type: ignore
        if img.shape[-1] == 1:
            img = img[:, :, 0]
        return img
    elif backend == "pillow":
        with io.BytesIO(content) as buff:
            img = Image.open(buff)
            img = _pillow2array(img, flag, channel_order)
        return img
    elif backend == "tifffile":
        with io.BytesIO(content) as buff:
            img = tifffile.imread(buff)
        return img
    else:
        img_np = np.frombuffer(content, np.uint8)
        flag = imread_flags[flag] if is_str(flag) else flag
        img = cv2.imdecode(img_np, flag)
        if flag == IMREAD_COLOR and channel_order == "rgb":
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        return img


@TRANSFORMS.register_module()
class LoadImageFromFile(BaseTransform):
    """从文件中加载图片数据
    - 需要输入含有图片路径的 key: `img_path`
    - 会将图片数据存储在 key: `img` 中, 并将图片的 shape 存储在 key: `img_shape` 中

    """

    def __init__(
        self,
        to_float32: bool = False,
        color_type: str = "color",
        decode_backend: str = "cv2",
        ignore_empty: bool = False,
        *,
        backend_args: Optional[dict] = None,
    ) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.decode_backend = decode_backend
        self.backend_args: Optional[dict] = None

        self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        filekeys = [key for key in results.keys() if "img_path" in key]
        for filekey in filekeys:
            filepath = results[filekey]
            try:
                img_bytes = fileio.get(filepath, backend_args=self.backend_args)
                img = imfrombytes(
                    img_bytes, flag=self.color_type, backend=self.decode_backend
                )
            except Exception as e:
                if self.ignore_empty:
                    return None
                else:
                    raise e

            assert img is not None, f"failed to load image: {filepath}"
            if self.to_float32:
                img = img.astype(np.float32)
            results[filekey.replace("img_path", "img_data")] = img
            results[filekey.replace("img_path", "img_shape")] = img.shape[:2]
            results[filekey.replace("img_path", "ori_shape")] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (
            f"{self.__class__.__name__}("
            f"ignore_empty={self.ignore_empty}, "
            f"to_float32={self.to_float32}, "
            f"color_type='{self.color_type}', "
            f"imdecode_backend='{self.decode_backend}', "
        )


        repr_str += f"backend_args={self.backend_args})"

        return repr_str


@TRANSFORMS.register_module()
class LoadMultiChannelImageFromFiles(BaseTransform):
    """Load multi-channel images from a list of separate channel files.
    - 需要输入含有图片路径列表的 key: `img_path`
    - 会将图片数据存储在 key: `img` 中, 并将图片的 shape 存储在 key: `img_shape` 中
    # TODO: 未测试
    """

    def __init__(
        self,
        to_float32: bool = False,
        color_type: str = "unchanged",
        imdecode_backend: str = "cv2",
        ignore_empty: bool = False,
        *,
        backend_args: dict = None,
    ) -> None:
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        self.backend_args = backend_args
        self.ignore_empty = ignore_empty

    def transform(self, results: dict) -> dict:
        """Transform functions to load multiple images and get images meta
        information.
        """
        filekeys = [key for key in results.keys() if "img_path" in key]
        for filekey in filekeys:
            img_paths = results[filekey]
            assert isinstance(img_paths, list)
            img = []
            for img_path in img_paths:
                try:
                    img_bytes = fileio.get(img_path, backend_args=self.backend_args)
                except Exception as e:
                    if self.ignore_empty:
                        return None
                    else:
                        raise e
                img.append(
                    imfrombytes(
                        img_bytes, flag=self.color_type, backend=self.imdecode_backend
                    )
                )
            img = np.stack(img, axis=-1)

            assert img is not None, f"failed to load image. "
            if self.to_float32:
                img = img.astype(np.float32)
            results[filekey.replace("img_path", "img_data")] = img
            results[filekey.replace("img_path", "img_shape")] = img.shape[:2]
            results[filekey.replace("img_path", "ori_shape")] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (
            f"{self.__class__.__name__}("
            f"to_float32={self.to_float32}, "
            f"color_type='{self.color_type}', "
            f"imdecode_backend='{self.imdecode_backend}', "
            f"backend_args={self.backend_args})"
        )
        return repr_str


audio_supported_backends = ["pydub", "librosa", "soundfile"]
audio_backend = "librosa"


def audiofrombytes(content: bytes, backend: Optional[str] = None) -> np.ndarray:
    """Load audio from bytes using different backends."""

    if backend is None:
        backend = audio_backend  # Default backend
    if backend not in audio_supported_backends:
        raise ValueError(
            f"Backend '{backend}' is not supported. Supported backends are: {audio_supported_backends}"
        )
    with io.BytesIO(content) as buff:
        if backend == "pydub":
            audio = AudioSegment.from_file(buff)
            sample_rate = audio.frame_rate
            audio = np.array(audio.get_array_of_samples())
            if audio.ndim == 1:
                audio = np.expand_dims(audio, axis=0)

            return audio, sample_rate
        else:
            audio, sample_rate = sf.read(buff, always_2d=True)
            return audio.T, sample_rate
        # else:
        #     audio, sample_rate = librosa.load(buff, sr=None)
        #     if audio.ndim == 1:
        #         audio = np.expand_dims(audio, axis=0)
        #     return audio, sample_rate


@TRANSFORMS.register_module()
class LoadAudioFromFile(BaseTransform):
    """从文件中加载音频数据
    - 需要输入含有音频路径的 key: `audio_path`
    - 会将音频数据存储在 key: `audio` 中, 并保存音频的采样率等信息.
    """

    def __init__(
        self,
        to_float32: bool = False,
        backend: str = "librosa",
        ignore_empty: bool = False,
        *,
        backend_args: Optional[dict] = None,
    ) -> None:
        self.backend_args: Optional[dict] = backend_args
        self.to_float32 = to_float32
        self.decode_backend = backend
        self.ignore_empty = ignore_empty

    def transform(self, results: dict) -> Optional[dict]:
        filekeys = [key for key in results.keys() if "audio_path" in key]
        for filekey in filekeys:
            filepath = results[filekey]
            try:
                audio_bytes = fileio.get(filepath, backend_args=self.backend_args)
                audio, sample_rate = audiofrombytes(
                    audio_bytes, backend=self.decode_backend
                )
            except Exception as e:
                if self.ignore_empty:
                    return None
                else:
                    raise e

            assert audio is not None, f"failed to load audio: {filepath}"
            if self.to_float32:
                audio = audio.astype(np.float32)
            results[filekey.replace("audio_path", "audio_data")] = audio
            results[filekey.replace("audio_path", "sample_rate")] = sample_rate
        return results

    def __repr__(self):
        repr_str = f"{self.__class__.__name__}("

        if self.backend_args is not None:
            repr_str += f"backend_args={self.backend_args})"

        return repr_str


@TRANSFORMS.register_module()
class LoadAudioFromFiles(BaseTransform):
    """从文件中加载音频数据
    - 需要输入含有音频路径列表的 key: `audio_path`
    - 会将音频数据存储在 key: `audio` 中, 并保存音频的采样率等信息.
    # TODO: 未测试
    """

    def __init__(
        self,
        to_float32: bool = False,
        decode_backend: str = "librosa",
        ignore_empty: bool = False,
        *,
        backend_args: Optional[dict] = None,
    ) -> None:
        self.backend_args: Optional[dict] = backend_args.copy()
        self.to_float32 = to_float32
        self.decode_backend = decode_backend
        self.ignore_empty = ignore_empty

    def transform(self, results: dict) -> Optional[dict]:
        assert isinstance(results["audio_path"], list)
        
        filekeys = [key for key in results.keys() if "audio_path" in key]
        for filekey in filekeys:
            audio = []
            sample_rate = []
            for filepath in results[filekey]:
                try:
                    audio_bytes = fileio.get(filepath, backend_args=self.backend_args)
                except Exception as e:
                    if self.ignore_empty:
                        return None
                    else:
                        raise e
                assert audio_bytes is not None, f"failed to load audio: {filepath}"
                audio_bytes = audiofrombytes(audio_bytes, backend=self.decode_backend)[0]
                if self.to_float32:
                    audio_bytes = audio_bytes.astype(np.float32)
                audio.append(audio_bytes)
                sample_rate.append(sample_rate)
            assert all(sample_rate) == sample_rate[0], "sample_rate must be the same"
            results[filekey.replace("audio_path", "audio_data")] = audio
            results[filekey.replace("audio_path", "sample_rate")] = sample_rate[0]
        return results

    def __repr__(self):
        repr_str = f"{self.__class__.__name__}("
        if self.backend_args is not None:
            repr_str += f"backend_args={self.backend_args})"
        return repr_str


video_supported_backends = ["pyav", "cv2", "decord"]

video_backend = "pyav"


def videofrombytes(
    content: bytes,
    backend: Optional[str] = None,
):
    """Load video from bytes using different backends."""

    if backend is None:
        backend = video_backend
    if backend not in video_supported_backends:
        raise ValueError(
            f"Backend '{backend}' is not supported. Supported backends are: {video_supported_backends}"
        )
    with io.BytesIO(content) as buff:
        if backend == "pyav":
            container = av.open(buff)
            video = []
            for frame in container.decode(video=0):
                video.append(frame.to_rgb().to_ndarray())
            return np.stack(video, axis=0), container.streams.video[0].average_rate
        elif backend == "decord":
            container = decord.VideoReader(buff)
            fps = container.get_avg_fps()
            video = container.get_batch(range(len(container)))
            return video, fps
        else:
            container = cv2.VideoCapture(buff)
            video = []
            while True:
                ret, frame = container.read()
                if not ret:
                    break
                video.append(frame)
            return np.stack(video, axis=0), container.get(cv2.CAP_PROP_FPS)


@TRANSFORMS.register_module()
class LoadVideoFromFile(BaseTransform):
    """从文件中加载视频数据
    - 需要输入含有视频路径的 key: `video_path`
    - 会将视频数据存储在 key: `video` 中, 并保存视频的帧率等信息.
    """

    def __init__(
        self,
        to_float32: bool = False,
        decode_backend: str = "pyav",
        ignore_empty: bool = False,
        *,
        backend_args: Optional[dict] = None,
    ) -> None:
        self.backend_args: Optional[dict] = backend_args.copy()
        self.to_float32 = to_float32
        self.decode_backend = decode_backend
        self.ignore_empty = ignore_empty

    def transform(self, results: dict) -> Optional[dict]:
        filekeys = [key for key in results.keys() if "video_path" in key]
        for filekey in filekeys:
            filepath = results[filekey]
            try:
                video_bytes = fileio.get(filepath, backend_args=self.backend_args)
                video, fps = videofrombytes(video_bytes, backend=self.decode_backend)
            except Exception as e:
                if self.ignore_empty:
                    return None
                else:
                    raise e

            assert video is not None, f"failed to load video: {filepath}"
            if self.to_float32:
                video = video.astype(np.float32)
            results[filekey.replace("video_path", "video_data")] = video
            results[filekey.replace("video_path", "fps")] = fps
        return results

    def __repr__(self):
        repr_str = f"{self.__class__.__name__}("
        if self.backend_args is not None:
            repr_str += f"backend_args={self.backend_args})"
        return repr_str
