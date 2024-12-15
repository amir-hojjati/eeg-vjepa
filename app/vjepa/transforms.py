# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from scipy.interpolate import interp1d

import torch
import numpy as np
import torchvision.transforms as transforms

import src.datasets.utils.video.transforms as video_transforms
from src.datasets.utils.video.randerase import RandomErasing


def make_transforms(
    random_horizontal_flip=True,
    random_resize_aspect_ratio=(3/4, 4/3),
    random_resize_scale=(0.3, 1.0),
    reprob=0.0,
    auto_augment=False,
    motion_shift=False,
    crop_size=224,
    normalize=((0.485, 0.456, 0.406),
               (0.229, 0.224, 0.225))
):

    _frames_augmentation = VideoTransform(
        random_horizontal_flip=random_horizontal_flip,
        random_resize_aspect_ratio=random_resize_aspect_ratio,
        random_resize_scale=random_resize_scale,
        reprob=reprob,
        auto_augment=auto_augment,
        motion_shift=motion_shift,
        crop_size=crop_size,
        normalize=normalize,
    )
    return _frames_augmentation


class VideoTransform(object):

    def __init__(
        self,
        random_horizontal_flip=True,
        random_resize_aspect_ratio=(3/4, 4/3),
        random_resize_scale=(0.3, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=224,
        normalize=((0.485, 0.456, 0.406),
                   (0.229, 0.224, 0.225)),
        spatial_aug=True,
        apply_augment=False
    ):

        # ***
        # # GLOBAL OVERRIDE
        crop_size = (19, 500)
        random_horizontal_flip = False
        spatial_aug = True
        apply_augment = False
        # ***
        if not isinstance(crop_size, tuple):
            crop_size = (crop_size, ) * 2

        self.random_horizontal_flip = random_horizontal_flip
        self.random_resize_aspect_ratio = random_resize_aspect_ratio
        self.random_resize_scale = random_resize_scale
        self.auto_augment = auto_augment
        self.spatial_aug = spatial_aug
        self.motion_shift = motion_shift
        self.crop_size = crop_size
        self.apply_augment = apply_augment
        self.mean = torch.tensor(normalize[0], dtype=torch.float32)
        self.std = torch.tensor(normalize[1], dtype=torch.float32)
        if not self.auto_augment:
            # Without auto-augment, PIL and tensor conversions simply scale uint8 space by 255.
            self.mean *= 255.
            self.std *= 255.

        self.autoaug_transform = video_transforms.create_random_augment(
            input_size=(crop_size[0], crop_size[1]),
            auto_augment='rand-m7-n4-mstd0.5-inc1',
            interpolation='bicubic',
        )

        self.spatial_transform = video_transforms.random_resized_crop_with_shift \
            if motion_shift else video_transforms.random_resized_crop

        self.reprob = reprob
        self.erase_transform = RandomErasing(
            reprob,
            mode='pixel',
            max_count=1,
            num_splits=1,
            device='cpu',
        )

    def __call__(self, buffer):

        if self.auto_augment:
            buffer = [transforms.ToPILImage()(frame) for frame in buffer]
            buffer = self.autoaug_transform(buffer)
            buffer = [transforms.ToTensor()(img) for img in buffer]
            buffer = torch.stack(buffer)  # T C H W
            buffer = buffer.permute(0, 2, 3, 1)  # T H W C
        else:
            buffer = torch.tensor(buffer, dtype=torch.float32)

        buffer = buffer.permute(3, 0, 1, 2)  # T H W C -> C T H W

        if self.spatial_aug:
            buffer = self.spatial_transform(
                images=buffer,
                target_height=self.crop_size[0],
                target_width=self.crop_size[1],
                scale=self.random_resize_scale,
                ratio=self.random_resize_aspect_ratio,
            )

        if self.apply_augment:
            buffer = augment(buffer)

        if self.random_horizontal_flip:
            buffer, _ = video_transforms.horizontal_flip(0.5, buffer)

        # buffer = _tensor_normalize_inplace(buffer, self.mean, self.std)
        if self.reprob > 0:
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = self.erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return buffer


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


def _tensor_normalize_inplace(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize (with dimensions C, T, H, W).
        mean (tensor): mean value to subtract (in 0 to 255 floats).
        std (tensor): std to divide (in 0 to 255 floats).
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()

    C, T, H, W = tensor.shape
    tensor = tensor.view(C, -1).permute(1, 0)  # Make C the last dimension
    tensor.sub_(mean).div_(std)
    tensor = tensor.permute(1, 0).view(C, T, H, W)  # Put C back in front
    return tensor


def augment(input_data):
    def add_noise(x, config_degree=0.05, degree=1.0):
        init_ch = 0
        init_epoch = 0
        degree = config_degree * degree
        # length of each channel
        ch_len = len(x[init_epoch][init_ch])
        # average channel range in first epoch
        ch_range = np.ptp(x[init_epoch], axis=1) + 1e-4  # add a small number for flat signal
        # calculate noises for each channel
        noise_high_frequency = degree * ch_range[np.newaxis].T * (2.0 * np.random.rand(ch_len) - 1)[np.newaxis]
        noise_low_frequency = degree * ch_range[np.newaxis].T * (2.0 * np.random.rand(ch_len // 100) - 1)[
            np.newaxis]
        x_old = np.linspace(0, 1, num=ch_len // 100, endpoint=True)
        x_new = np.linspace(0, 1, num=ch_len, endpoint=True)
        interpolation = interp1d(x_old, noise_low_frequency, kind="linear")
        noise2 = interpolation(x_new)
        epochs = []
        for epoch in range(x.shape[0]):
            channels = []
            for ch in range(x.shape[1]):
                out_x = x[epoch][ch] + noise2[ch] + noise_high_frequency[ch]
                channels.append(out_x)
            epochs.append(torch.stack(channels))
        return torch.stack(epochs)

    def scaling(x, config_degree=0.05, noise_degree=2.0):
        degree = config_degree * (noise_degree + np.random.rand())
        factor = 2.0 * np.random.normal(size=x.shape[2]) - 1
        factor = 1.5 + (2.0 * np.random.rand()) + degree * factor
        epochs = []
        for epoch in range(x.shape[0]):
            ret = np.zeros_like(x[epoch])
            for ch in range(x.shape[1]):
                ret[ch] = x[epoch][ch] * factor
            epochs.append(torch.from_numpy(ret))
        return torch.stack(epochs)

    if not torch.is_tensor(input_data):
        input_data = torch.from_numpy(input_data)
    input_data = input_data.squeeze()
    return np.float32(scaling(add_noise(input_data)).unsqueeze(0).numpy())
