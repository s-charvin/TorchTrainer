
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import copy
import random
from pathlib import Path

import io
import numpy as np
import os.path as osp
import cv2
import torch

try:
    from PIL import Image
except ImportError:
    Image = None


import librosa

try:
    import torchaudio
except ImportError:
    torchaudio = None

from .base import BaseTransform
from TorchTrainer.utils.registry import TRANSFORMS
import TorchTrainer.utils.fileio as fileio
from TorchTrainer.utils import is_filepath, is_str

Number = Union[int, float]
imread_backend = "cv2"

cv2_interp_codes = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}


cv2_border_modes = {
    "constant": cv2.BORDER_CONSTANT,
    "replicate": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_REFLECT,
    "wrap": cv2.BORDER_WRAP,
    "reflect_101": cv2.BORDER_REFLECT_101,
    "transparent": cv2.BORDER_TRANSPARENT,
    "isolated": cv2.BORDER_ISOLATED,
}

if Image is not None:
    if hasattr(Image, "Resampling"):
        pillow_interp_codes = {
            "nearest": Image.Resampling.NEAREST,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "box": Image.Resampling.BOX,
            "lanczos": Image.Resampling.LANCZOS,
            "hamming": Image.Resampling.HAMMING,
        }
    else:
        pillow_interp_codes = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "box": Image.BOX,
            "lanczos": Image.LANCZOS,
            "hamming": Image.HAMMING,
        }


def _scale_size(
    size: Tuple[int, int],
    scale: Union[float, int, Tuple[float, float], Tuple[int, int]],
) -> Tuple[int, int]:
    """Rescale a size by a ratio."""
    if isinstance(scale, (float, int)):
        scale = (scale, scale)
    w, h = size
    return int(w * float(scale[0]) + 0.5), int(h * float(scale[1]) + 0.5)


def imresize(
    img: np.ndarray,
    size: Tuple[int, int],
    return_scale: bool = False,
    interpolation: str = "bilinear",
    out: Optional[np.ndarray] = None,
    backend: Optional[str] = None,
) -> Union[Tuple[np.ndarray, float, float], np.ndarray]:
    """Resize image to a given size."""
    h, w = img.shape[:2]
    if backend is None:
        backend = imread_backend
    if backend not in ["cv2", "pillow"]:
        raise ValueError(
            f"backend: {backend} is not supported for resize."
            f"Supported backends are 'cv2', 'pillow'"
        )

    if backend == "pillow":
        assert img.dtype == np.uint8, "Pillow backend only support uint8 type"
        pil_image = Image.fromarray(img)
        pil_image = pil_image.resize(size, pillow_interp_codes[interpolation])
        resized_img = np.array(pil_image)
    else:
        resized_img = cv2.resize(
            img, size, dst=out, interpolation=cv2_interp_codes[interpolation]
        )
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale


def rescale_size(
    old_size: tuple,
    scale: Union[float, int, Tuple[int, int]],
    return_scale: bool = False,
) -> tuple:
    """Calculate the new size to be rescaled to."""
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f"Invalid scale {scale}, must be positive.")
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
    else:
        raise TypeError(
            f"Scale must be a number or tuple of int, but got {type(scale)}"
        )

    new_size = _scale_size((w, h), scale_factor)

    if return_scale:
        return new_size, scale_factor
    else:
        return new_size


def imrescale(
    img: np.ndarray,
    scale: Union[float, int, Tuple[int, int]],
    return_scale: bool = False,
    interpolation: str = "bilinear",
    backend: Optional[str] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """Resize image while keeping the aspect ratio."""
    h, w = img.shape[:2]
    new_size, scale_factor = rescale_size((w, h), scale, return_scale=True)
    rescaled_img = imresize(img, new_size, interpolation=interpolation, backend=backend)
    if return_scale:
        return rescaled_img, scale_factor
    else:
        return rescaled_img


def imnormalize(img, mean, std, to_rgb=True):
    """Normalize an image with mean and std."""
    img = img.copy().astype(np.float32)
    return imnormalize_(img, mean, std, to_rgb)


def imnormalize_(img, mean, std, to_rgb=True):
    """Inplace normalize an image with mean and std."""
    # cv2 inplace normalization does not accept uint8
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img


@TRANSFORMS.register_module()
class NormalizeImage(BaseTransform):
    """Normalize the image."""

    def __init__(
        self,
        mean: Sequence[Number],
        std: Sequence[Number],
        to_rgb: bool = True,
        keys: Union[str, Sequence[str]] = None,
    ) -> None:
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        if keys is not None:
            self.keys = keys if isinstance(keys, Sequence) else [keys]
        else:
            self.keys = None
        self.prefix = "img_data"

    def transform(self, results: dict) -> dict:
        """Function to normalize images."""
        if self.keys is not None:
            pass
        else:
            self.keys = [key for key in results.keys() if (self.prefix in key)]

        for key in self.keys:
            if results.get(key, None) is None:
                raise KeyError(f"Can not find key {key}")
            results[key] = imnormalize(results[key], self.mean, self.std, self.to_rgb)
            results[f"norm_{key}_cfg"] = dict(
                mean=self.mean, std=self.std, to_rgb=self.to_rgb
            )
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"
        return repr_str


@TRANSFORMS.register_module()
class NormalizeVideo(BaseTransform):
    """Normalize the video."""

    def __init__(
        self,
        mean: Sequence[Number],
        std: Sequence[Number],
        to_rgb: bool = True,
        keys: Union[str, Sequence[str]] = None,
    ) -> None:
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        if keys is not None:
            self.keys = keys if isinstance(keys, Sequence) else [keys]
        else:
            self.keys = None
        self.prefix = "video_data"

    def transform(self, results: dict) -> dict:
        """Function to normalize video."""
        if self.keys is not None:
            pass
        else:
            self.keys = [key for key in results.keys() if (self.prefix in key)]

        for key in self.keys:
            data = results[key]
            for j in range(len(data)):
                data[j] = imnormalize(data[j], self.mean, self.std, self.to_rgb)
            results[key] = data
            results[f"norm_{key}_cfg"] = dict(
                mean=self.mean, std=self.std, to_rgb=self.to_rgb
            )
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"
        return repr_str


@TRANSFORMS.register_module()
class ResizeImage(BaseTransform):
    def __init__(
        self,
        scale: Optional[Union[int, Tuple[int, int]]] = None,
        scale_factor: Optional[Union[float, Tuple[float, float]]] = None,
        keep_ratio: bool = False,
        clip_object_border: bool = True,
        backend: str = "cv2",
        interpolation="bilinear",
        keys: Union[str, Sequence[str]] = None,
    ) -> None:
        assert scale is not None or scale_factor is not None, (
            "`scale` and" "`scale_factor` can not both be `None`"
        )
        if scale is None:
            self.scale = None
        else:
            if isinstance(scale, int):
                self.scale = (scale, scale)
            else:
                self.scale = scale

        self.backend = backend
        self.interpolation = interpolation
        self.keep_ratio = keep_ratio
        self.clip_object_border = clip_object_border
        if scale_factor is None:
            self.scale_factor = None
        elif isinstance(scale_factor, float):
            self.scale_factor = (scale_factor, scale_factor)
        elif isinstance(scale_factor, tuple):
            assert (len(scale_factor)) == 2
            self.scale_factor = scale_factor
        else:
            raise TypeError(
                f"expect scale_factor is float or Tuple(float), but"
                f"get {type(scale_factor)}"
            )
        if keys is not None:
            self.keys = keys if isinstance(keys, Sequence) else [keys]
        else:
            self.keys = None

        self.prefix = "img_data"

    def _resize_img(self, results: dict) -> None:
        """Resize images with ``results['scale']``."""
        if self.keys is not None:
            pass
        else:
            self.keys = [key for key in results.keys() if (self.prefix in key)]

        for key in self.keys:
            if results.get(key, None) is None:
                raise KeyError(f"Can not find key {key}")
            data = results[key]

            if self.keep_ratio:
                results[key], scale_factor = imrescale(
                    data,
                    results["scale"],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend,
                )
                new_h, new_w = results[key].shape[:2]
                h, w = data.shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                results[key], w_scale, h_scale = imresize(
                    data,
                    results["scale"],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend,
                )

            results[f"shape_{key}"] = data[0].shape[:2]
            results[f"scale_factor_{key}"] = (w_scale, h_scale)
            results[f"keep_ratio_{key}"] = self.keep_ratio

    def transform(self, results: dict) -> dict:
        """Transform function to resize images."""

        if self.scale:
            results["scale"] = self.scale
        else:
            img_shape = results["img"].shape[:2]
            results["scale"] = _scale_size(img_shape[::-1], self.scale_factor)

        self._resize_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(scale={self.scale}, "
        repr_str += f"scale_factor={self.scale_factor}, "
        repr_str += f"keep_ratio={self.keep_ratio}, "
        repr_str += f"clip_object_border={self.clip_object_border}), "
        repr_str += f"backend={self.backend}), "
        repr_str += f"interpolation={self.interpolation})"
        return repr_str


@TRANSFORMS.register_module()
class ResizeVideo(BaseTransform):
    def __init__(
        self,
        scale: Optional[Union[int, Tuple[int, int]]] = None,
        scale_factor: Optional[Union[float, Tuple[float, float]]] = None,
        keep_ratio: bool = False,
        clip_object_border: bool = True,
        backend: str = "cv2",
        interpolation="bilinear",
        keys: Union[str, Sequence[str]] = None,
    ) -> None:
        assert scale is not None or scale_factor is not None, (
            "`scale` and" "`scale_factor` can not both be `None`"
        )
        if scale is None:
            self.scale = None
        else:
            if isinstance(scale, int):
                self.scale = (scale, scale)
            else:
                self.scale = scale

        self.backend = backend
        self.interpolation = interpolation
        self.keep_ratio = keep_ratio
        self.clip_object_border = clip_object_border
        if scale_factor is None:
            self.scale_factor = None
        elif isinstance(scale_factor, float):
            self.scale_factor = (scale_factor, scale_factor)
        elif isinstance(scale_factor, tuple):
            assert (len(scale_factor)) == 2
            self.scale_factor = scale_factor
        else:
            raise TypeError(
                f"expect scale_factor is float or Tuple(float), but"
                f"get {type(scale_factor)}"
            )
        if keys is not None:
            self.keys = keys if isinstance(keys, Sequence) else [keys]
        else:
            self.keys = None
        self.prefix = "video_data"

    def _resize_video(self, results: dict) -> None:
        """Resize video with ``results['scale']``."""

        if self.keys is not None:
            pass
        else:
            self.keys = [key for key in results.keys() if (self.prefix in key)]
        for key in self.keys:
            if results.get(key, None) is None:
                raise KeyError(f"Can not find key {key}")
            data = results[key]

            for j in range(len(data)):
                if self.keep_ratio:
                    img, scale_factor = imrescale(
                        data[j],
                        results["scale"],
                        interpolation=self.interpolation,
                        return_scale=True,
                        backend=self.backend,
                    )
                    new_h, new_w = img.shape[:2]
                    h, w = data[j].shape[:2]
                    w_scale = new_w / w
                    h_scale = new_h / h
                else:
                    img, w_scale, h_scale = imresize(
                        data[j],
                        results["scale"],
                        interpolation=self.interpolation,
                        return_scale=True,
                        backend=self.backend,
                    )
                data[j] = img

            results[key] = data
            results[f"shape_{key}"] = results[key][0].shape[1:3]
            results[f"scale_factor_{key}"] = (w_scale, h_scale)
            results[f"keep_ratio_{key}"] = self.keep_ratio

    def transform(self, results: dict) -> dict:
        """Transform function to resize images."""

        if self.scale:
            results["scale"] = self.scale
        else:
            img_shape = results["img"].shape[:2]
            results["scale"] = _scale_size(img_shape[::-1], self.scale_factor)
        self._resize_video(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(scale={self.scale}, "
        repr_str += f"scale_factor={self.scale_factor}, "
        repr_str += f"keep_ratio={self.keep_ratio}, "
        repr_str += f"clip_object_border={self.clip_object_border}), "
        repr_str += f"backend={self.backend}), "
        repr_str += f"interpolation={self.interpolation})"
        return repr_str


audio_backend = "librosa"


def get_window(name):
    if name == "hann":
        return torch.hann_window
    elif name == "hamming":
        return torch.hamming_window
    elif name == "blackman":
        return torch.blackman_window
    elif name == "kaiser":
        return torch.kaiser_window
    else:
        raise ValueError(f"Unsupported window type: {name}")


def extract_mfcc(
    audio: np.ndarray,
    sample_rate: int,
    n_mfcc=40,
    dct_type=2,
    backend: Optional[str] = None,
    **kwargs,
):
    """Extract MFCC features from audio."""
    n_fft = kwargs.get("n_fft", 1024)
    win_length = kwargs.get("win_length", 1024)
    hop_length = kwargs.get("hop_length", 256)
    window = kwargs.get("window", "hann")
    center = kwargs.get("center", True)
    pad_mode = kwargs.get("pad_mode", "reflect")
    power = kwargs.get("power", 2.0)
    n_mels = kwargs.get("n_mels", 128)
    f_min = kwargs.get("f_min", 80.0)
    f_max = kwargs.get("f_max", 7600.0)
    window_fn = kwargs.get("window_fn", get_window(window))

    lifter = kwargs.get("lifter", 0)
    log_mels = kwargs.get("log_mels", True)
    pad = kwargs.get("pad", 0)

    normalized = kwargs.get("normalized", True)
    onesided = kwargs.get("onesided", True)

    if backend is None:
        backend = audio_backend
    if backend not in ["librosa", "torchaudio"]:
        raise ValueError(
            f"backend: {backend} is not supported for extract mfcc."
            f"Supported backends are 'librosa'"
        )
    assert sample_rate is not None, "sample_rate is None"
    if backend == "librosa":
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=n_mfcc,
            dct_type=dct_type,
            lifter=lifter,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            power=power,
            n_mels=n_mels,
            fmin=f_min,
            fmax=f_max,
        )
        mfcc = np.transpose(mfcc, (0, 2, 1))
    elif backend == "torchaudio":
        audio = torch.from_numpy(audio.astype(np.float32))
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            dct_type=dct_type,
            n_mfcc=n_mfcc,
            norm="ortho",
            log_mels=log_mels,
            melkwargs=dict(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                f_min=f_min,
                f_max=f_max,
                pad=pad,
                n_mels=n_mels,
                window_fn=window_fn,
                power=power,
                normalized=normalized,
                center=center,
                onesided=onesided,
                norm="slaney",
            ),
        )(audio)
        mfcc = mfcc.permute(0, 2, 1).numpy()  # 1, t, n_mfcc
    else:
        raise ValueError(f"Unsupported audio backend: {backend}")
    return mfcc


@TRANSFORMS.register_module()
class MFCC(BaseTransform):
    def __init__(
        self,
        backend: Optional[str] = None,
        override=True,
        keys: Union[str, Sequence[str]] = None,
        **kwargs,
    ) -> None:
        self.kwargs = kwargs
        self.backend = backend
        self.override = override
        if keys is not None:
            self.keys = keys if isinstance(keys, Sequence) else [keys]
        else:
            self.keys = None
        self.prefix = "audio_data"

    def transform(self, results: dict) -> dict:
        """Transform function to extract mfcc features."""
        if self.keys is not None:
            pass
        else:
            self.keys = [key for key in results.keys() if (self.prefix in key)]
        sample_rate = results.get("sample_rate", None)

        for key in self.keys:
            if results.get(key, None) is None:
                raise KeyError(f"Can not find key {key}")
            new_key = key.replace("audio_data", "mfcc_data")
            results[new_key] = extract_mfcc(
                results[key],
                sample_rate=sample_rate,
                backend=self.backend,
                **self.kwargs,
            )
            if self.override:
                results.pop(key, None)
            results[f"mfcc_cfg_{key}"] = self.kwargs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(backend='{self.backend}', mfcc_cfg='{self.kwargs}')"
        return repr_str


def extract_fbank(audio: np.ndarray, backend: Optional[str] = None, **kwargs):
    """Extract fbank features from audio."""
    if backend is None:
        backend = audio_backend
    if backend not in ["torchaudio"]:
        raise ValueError(
            f"backend: {backend} is not supported for extract fbank."
            f"Supported backends are 'torchaudio'"
        )
    if backend == "torchaudio":
        fbank = torchaudio.compliance.kaldi.fbank(audio, **kwargs)
    else:
        raise ValueError(f"Unsupported audio backend: {backend}")
    return fbank


@TRANSFORMS.register_module()
class Fbank(BaseTransform):
    def __init__(
        self,
        backend: Optional[str] = None,
        override=True,
        keys: Union[str, Sequence[str]] = None,
        **kwargs,
    ) -> None:
        self.kwargs = kwargs
        self.backend = backend
        self.override = override
        if keys is not None:
            self.keys = keys if isinstance(keys, Sequence) else [keys]
        else:
            self.keys = None
        self.prefix = "audio_data"

    def transform(self, results: dict) -> dict:
        """Transform function to extract fbank features."""
        if self.keys is not None:
            pass
        else:
            self.keys = [key for key in results.keys() if (self.prefix in key)]
        for key in self.keys:
            if results.get(key, None) is None:
                raise KeyError(f"Can not find key {key}")
            new_key = key.replace("audio_data", "fbank_data")
            results[new_key] = extract_fbank(results[key], self.backend, **self.kwargs)
            if self.override:
                results.pop(key, None)
            results[f"fbank_cfg_{key}"] = self.kwargs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(backend='{self.backend}', fbank_cfg = '{self.kwargs}')"
        return repr_str


def extract_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = 1024,
    win_length: int = 1024,
    hop_length: int = 256,
    window: str = "hann",
    backend: Optional[str] = None,
    **kwargs,
):
    """Extract mel spectrogram features from audio."""
    n_fft = n_fft
    win_length = win_length
    hop_length = hop_length
    window = window

    center = kwargs.get("center", True)
    pad_mode = kwargs.get("pad_mode", "reflect")
    power = kwargs.get("power", 2.0)
    n_mels = kwargs.get("n_mels", 128)
    f_min = kwargs.get("f_min", 80.0)
    f_max = kwargs.get("f_max", 7600.0)

    window_fn = kwargs.get("window_fn", get_window(window))

    normalized = kwargs.get("normalized", True)
    onesided = kwargs.get("onesided", True)
    pad = kwargs.get("pad", 0)

    if backend is None:
        backend = audio_backend
    if backend not in ["torchaudio", "librosa"]:
        raise ValueError(
            f"backend: {backend} is not supported for extract mel spectrogram."
            f"Supported backends are 'torchaudio'"
        )
    if backend == "torchaudio":
        audio = torch.from_numpy(audio.astype(np.float32))
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            pad=pad,
            n_mels=n_mels,
            window_fn=window_fn,
            power=power,
            normalized=normalized,
            center=center,
            onesided=onesided,
            pad_mode=pad_mode,
            norm="slaney",
        )(audio)
        mel_spectrogram = mel_spectrogram.permute(0, 2, 1).numpy()  # 1, t, n_mels
    elif backend == "librosa":
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            power=power,
            n_mels=n_mels,
            fmin=f_min,
            fmax=f_max,
        )
        mel_spectrogram = np.transpose(mel_spectrogram, (0, 2, 1))

    return mel_spectrogram


@TRANSFORMS.register_module()
class MelSpectrogram(BaseTransform):
    def __init__(
        self,
        backend: Optional[str] = None,
        override=True,
        keys: Union[str, Sequence[str]] = None,
        **kwargs,
    ) -> None:
        self.kwargs = kwargs
        self.backend = backend
        self.override = override
        if keys is not None:
            self.keys = keys if isinstance(keys, Sequence) else [keys]
        else:
            self.keys = None
        self.prefix = "audio_data"

    def transform(self, results: dict) -> dict:
        """Transform function to extract mel spectrogram features."""
        if self.keys is not None:
            pass
        else:
            self.keys = [key for key in results.keys() if (self.prefix in key)]
        sample_rate = results.get("sample_rate", None)
        for key in self.keys:
            if results.get(key, None) is None:
                raise KeyError(f"Can not find key {key}")

            new_key = key.replace("audio_data", "mel_spectrogram_data")

            results[new_key] = extract_mel_spectrogram(
                audio=results[key],
                sample_rate=sample_rate,
                backend=self.backend,
                **self.kwargs,
            )
            if self.override:
                results.pop(key, None)
            results[f"mel_spectrogram_cfg_{key}"] = self.kwargs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(backend='{self.backend}', mel_spectrogram_cfg='{self.kwargs}')"
        return repr_str


@TRANSFORMS.register_module()
class CropFrames(BaseTransform):
    def __init__(
        self,
        sample_num: int,
        mode: str = "center",
        backend: Optional[str] = None,
        keys: Union[str, Sequence[str]] = None,
        **kwargs,
    ) -> None:
        assert sample_num > 0

        self.sample_num = sample_num

        self.kwargs = kwargs
        self.backend = backend
        self.mode = mode
        if keys is not None:
            self.keys = keys if isinstance(keys, Sequence) else [keys]
        else:
            self.keys = None
        self.prefix = "video_data"

    def transform(self, results: dict) -> dict:
        """Transform function to sample frames."""
        if self.keys is not None:
            pass
        else:
            self.keys = [key for key in results.keys() if (self.prefix in key)]

        for key in self.keys:
            if results.get(key, None) is None:
                raise KeyError(f"Can not find key {key}")

            video_len = len(results[key])
            if video_len > self.sample_num:
                if self.mode == "random":
                    start = np.random.randint(0, video_len - self.sample_num)
                elif self.mode == "center":
                    start = (video_len - self.sample_num) // 2
                else:
                    raise ValueError(f"Unsupported mode: {self.mode}")
            else:
                start = 0
            results[key] = results[key][start : start + self.sample_num]
        results[f"crop_frames_cfg_{key}"] = dict(
            sample_num=self.sample_num, mode=self.mode
        )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__

        repr_str += f"(backend='{self.backend}', sample_num={self.sample_num}, mode='{self.mode}')"
        return repr_str


@TRANSFORMS.register_module()
class ResampleAudio(BaseTransform):
    def __init__(
        self,
        sample_rate,
        input_sample_rate=None,
        backend: Optional[str] = None,
        keys: Union[str, Sequence[str]] = None,
        **kwargs,
    ) -> None:
        self.kwargs = kwargs
        self.backend = backend
        self.output_sample_rate = sample_rate
        self.input_sample_rate = input_sample_rate
        if keys is not None:
            self.keys = keys if isinstance(keys, Sequence) else [keys]
        else:
            self.keys = None
        self.prefix = "audio_data"

    def transform(self, results: dict) -> dict:
        """Transform function to sample audio features."""
        if self.keys is not None:
            pass
        else:
            self.keys = [key for key in results.keys() if (self.prefix in key)]
        if self.input_sample_rate is None:
            self.input_sample_rate = results.get("sample_rate", None)
        assert self.input_sample_rate is not None, " sample_rate is None"

        for key in self.keys:
            if results.get(key, None) is None:
                raise KeyError(f"Can not find key {key}")

            results[key] = librosa.resample(
                results[key],
                orig_sr=self.input_sample_rate,
                target_sr=self.output_sample_rate,
            )

        results[f"resample_audio_cfg_{key}"] = dict(
            input_sample_rate=self.input_sample_rate,
            output_sample_rate=self.output_sample_rate,
        )
        results["sample_rate"] = self.output_sample_rate
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(backend='{self.backend}', interval={self.interval})"
        return repr_str


@TRANSFORMS.register_module()
class ResampleVideo(BaseTransform):
    def __init__(
        self,
        in_fps,
        out_fps,
        backend: Optional[str] = None,
        keys: Union[str, Sequence[str]] = None,
        **kwargs,
    ) -> None:
        self.kwargs = kwargs
        self.backend = backend
        self.input_fps = in_fps
        self.output_fps = out_fps
        if keys is not None:
            self.keys = keys if isinstance(keys, Sequence) else [keys]
        else:
            self.keys = None
        self.prefix = "video_data"

    def transform(self, results: dict) -> dict:
        """Transform function to sample frames."""
        assert self.input_fps != results.get(
            "fps", self.input_fps
        ), "The specified fps does not match the input reserved fps"
        if self.keys is not None:
            pass
        else:
            self.keys = [key for key in results.keys() if (self.prefix in key)]
        for key in self.keys:
            if results.get(key, None) is None:
                raise KeyError(f"Can not find key {key}")

            frame_count, height, width, _ = results[key].shape
            output_frame_count = int(frame_count * self.output_fps / self.input_fps)
            results[key] = cv2.resize(
                results[key],
                (width, height),
                fx=output_frame_count / frame_count,
                fy=1.0,
                interpolation=cv2.INTER_LINEAR,
            )

        results[f"resample_video_cfg_{key}"] = dict(
            input_fps=self.input_fps, output_fps=self.output_fps
        )
        results["fps"] = self.output_fps
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(backend='{self.backend}', interval={self.interval})"
        return repr_str


@TRANSFORMS.register_module()
class TimeRandomSplit(BaseTransform):
    def __init__(
        self,
        win_length: int,
        hop_length: int = None,
        backend: Optional[str] = None,
        override=True,
        keys: Union[str, Sequence[str]] = None,
        prefix=None,
        **kwargs,
    ) -> None:
        assert bool(keys) or bool(prefix), "keys and prefix can not both be None"
        assert win_length > 0
        self.kwargs = kwargs
        self.backend = backend
        self.override = override
        self.win_length = win_length
        if hop_length is None:
            hop_length = win_length
        self.hop_length = hop_length
        if keys is not None:
            self.keys = keys if isinstance(keys, Sequence) else [keys]
        else:
            self.keys = None
        self.prefix = prefix if prefix is not None else ""

    def transform(self, results: dict) -> dict:
        """Transform function to split data."""
        if self.keys is not None:
            pass
        else:
            self.keys = [key for key in results.keys() if (self.prefix in key)]
        for key in self.keys:
            if results.get(key, None) is None:
                raise KeyError(f"Can not find key {key}")

            data_len = results[key].shape[1]
            total_nums = (
                int(np.ceil((data_len - self.win_length) / self.hop_length)) + 1
            )

            if total_nums > 1:
                # 随机选择一个片段
                num = np.random.randint(0, total_nums - 1)
                start = num * self.hop_length
                end = min(start + self.win_length, data_len)
                results[key] = results[key][:, start:end]

            elif total_nums == 1:
                results[key] = np.pad(
                    results[key],
                    ((0, 0), (0, self.win_length - data_len)),
                    mode="constant",
                    constant_values=0,
                )

            results[f"split_{key}_cfg"] = dict(
                win_length=self.win_length, hop_length=self.hop_length
            )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(backend='{self.backend}', win_length={self.win_length}, hop_length={self.hop_length})"
        return repr_str


@TRANSFORMS.register_module()
class PadCutAudio(BaseTransform):
    def __init__(
        self,
        sample_num: int,
        keys: Union[str, Sequence[str]] = None,
        pad_mode: str = "center",  # "center", "left", "right", "random"
        mode: str = "constant",
        value: float = 0.0,
    ) -> None:
        self.sample_num = int(sample_num)
        self.pad_mode = pad_mode
        self.mode = mode
        self.value = value
        if keys is not None:
            self.keys = keys if isinstance(keys, Sequence) else [keys]
        else:
            self.keys = None
        self.prefix = "audio_data"

    def transform(self, results: dict) -> dict:
        """Transform function to pad audio."""
        if self.keys is not None:
            pass
        else:
            self.keys = [key for key in results.keys() if (self.prefix in key)]
        for key in self.keys:
            if results.get(key, None) is None:
                raise KeyError(f"Can not find key {key}")

            data_len = results[key].shape[1]
            if data_len < self.sample_num:
                if self.pad_mode == "center":
                    pad_left = (self.sample_num - data_len) // 2
                    pad_right = self.sample_num - data_len - pad_left
                elif self.pad_mode == "left":
                    pad_left = self.sample_num - data_len
                    pad_right = 0
                elif self.pad_mode == "right":
                    pad_left = 0
                    pad_right = self.sample_num - data_len
                elif self.pad_mode == "random":
                    pad_left = np.random.randint(0, self.sample_num - data_len)
                    pad_right = self.sample_num - data_len - pad_left
                else:
                    raise ValueError(f"Unsupported pad_mode: {self.pad_mode}")
                results[key] = np.pad(
                    results[key],
                    ((0, 0), (pad_left, pad_right)),
                    mode=self.mode,
                    constant_values=self.value,
                )
            elif data_len > self.sample_num:
                if self.pad_mode == "center":
                    cut_left = (data_len - self.sample_num) // 2
                    cut_right = data_len - self.sample_num - cut_left
                elif self.pad_mode == "left":
                    cut_left = data_len - self.sample_num
                    cut_right = 0
                elif self.pad_mode == "right":
                    cut_left = 0
                    cut_right = data_len - self.sample_num
                elif self.pad_mode == "random":
                    cut_left = np.random.randint(0, data_len - self.sample_num)
                    cut_right = data_len - self.sample_num - cut_left
                else:
                    raise ValueError(f"Unsupported pad_mode: {self.pad_mode}")
                results[key] = results[key][:, cut_left : data_len - cut_right]

            results[f"pad_{key}_cfg"] = dict(
                padding=self.pad_mode, mode=self.mode, value=self.value
            )
        return results
