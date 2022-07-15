import math
import random
import logging

import numpy as np
from scipy.ndimage.filters import gaussian_filter
import torch
from resize_right import resize
from torch import nn
import torchvision.transforms as T
from torch.nn import functional as F
import torchvision.transforms.functional as TF
from PIL import ImageDraw


logger = logging.getLogger(__name__)


def save_inner_cut_bounds_image(input, bounds_list, f_name):
    image = TF.to_pil_image(input.clamp(0, 1).squeeze(0))
    draw = ImageDraw.Draw(image)
    for points in bounds_list:
        # Bounds are in form left, right, top, bottom.
        # Convert that here to top left x and y, bottom right x and y.
        draw.rectangle(
            (points[0], points[2], points[1], points[3]),
            outline=(255, 0, 0)
        )
    image.save(
        f_name, quality=99
    )


def save_cut_image(cutout, f_name: str):
    TF.to_pil_image(cutout.clamp(0, 1).squeeze(0)).save(
        f_name, quality=99
    )


def sinc(x):
    return torch.where(x != 0,
                       torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x / a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.reshape([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.reshape([n, c, h, w])
    return F.interpolate(input,
                         size,
                         mode='bicubic',
                         align_corners=align_corners)


def center_to_bounds(center_x, center_y, cut_size, image_x, image_y):
    pad_size = int(cut_size / 2)
    left_bound = max((center_x - pad_size), 0)
    right_bound = min((center_x + pad_size), image_x)
    top_bound = max((center_y - pad_size), 0)
    bottom_bound = min((center_y + pad_size), image_y)
    return left_bound, right_bound, top_bound, bottom_bound


def random_sample(side_x, side_y, inner_mask_size=0):
    return (
        torch.randint(inner_mask_size, side_x - inner_mask_size, ()),
        torch.randint(inner_mask_size, side_y - inner_mask_size, ())
    )


class CutHeatmap(object):

    def __init__(
            self,
            side_x,
            side_y,
            decay_scale=0.3,
            decay_gaussian_sigma=7,
            overlap_penalty_coef=0.3
    ):
        self.heatmap = np.ones((side_y, side_x), dtype=np.float32)
        self.decay_scale = decay_scale
        self.decay_gaussian_sigma = decay_gaussian_sigma
        self.overlap_penalty_coef = overlap_penalty_coef

    def add_cut(self, center_x, center_y, cut_size):
        left, right, top, bottom = center_to_bounds(
            center_x,
            center_y,
            cut_size,
            image_x=self.heatmap.shape[-1],
            image_y=self.heatmap.shape[-2]
        )
        self.heatmap[
            top:bottom,
            left:right
        ] *= self.overlap_penalty_coef

    def decay(self):
        self.heatmap = gaussian_filter(self.heatmap, sigma=self.decay_gaussian_sigma)
        self.heatmap = (
            np.ones(self.heatmap.shape, dtype=np.float32) * self.decay_scale + self.heatmap
        ) / (1 + self.decay_scale)

    def sample_centerpoint(self, cut_size, padded=False):
        cut_offset = int(cut_size / 2)
        if padded:
            # If we will pad the image, any centerpoint is valid
            centerpoints = self.heatmap
        else:
            # Otherwise, only select points half a cut size away from the edge
            centerpoints = self.heatmap[cut_offset:-cut_offset, cut_offset:-cut_offset]
        linear_idx = np.random.choice(centerpoints.size, p=centerpoints.ravel() / float(centerpoints.sum()))
        y, x = np.unravel_index(linear_idx, centerpoints.shape)
        if not padded:
            x += cut_offset
            y += cut_offset
        return x, y

    def to_image(self):
        return TF.to_pil_image(self.heatmap * 255).convert("RGB")

    def save_image(self, name="cut_heatmap.jpg"):
        self.to_image().save(name, quality=99)


class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, skip_augs=False):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.skip_augs = skip_augs
        self.augs = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomPerspective(distortion_scale=0.4, p=0.7),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomGrayscale(p=0.15),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ])

    def forward(self, input):
        input = T.Pad(input.shape[2] // 4, fill=0)(input)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)

        cutouts = []
        for ch in range(self.cutn):
            if ch > self.cutn - self.cutn // 4:
                cutout = input.clone()
            else:
                size = int(max_size * torch.zeros(1, ).normal_(
                    mean=.8, std=.3).clip(float(self.cut_size / max_size), 1.))
                offsetx = torch.randint(0, abs(sideX - size + 1), ())
                offsety = torch.randint(0, abs(sideY - size + 1), ())
                cutout = input[:, :, offsety:offsety + size,
                         offsetx:offsetx + size]

            if not self.skip_augs:
                cutout = self.augs(cutout)
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
            del cutout

        cutouts = torch.cat(cutouts, dim=0)
        # For parity with MakeCutoutsDango, return an empty innercut bound list
        return cutouts, []


class MakeCutoutsDango(nn.Module):
    def __init__(self,
                 cut_size,
                 Overview=4,
                 InnerCrop=0,
                 IC_Size_Pow=0.5,
                 IC_Grey_P=0.2,
                 animation_mode='None'
    ):
        super().__init__()
        self.cut_size = cut_size
        self.overview_cut_count = Overview
        self.inner_cut_count = InnerCrop
        self.inner_cut_size_exponent = IC_Size_Pow
        self.inner_cut_grey_exponent = IC_Grey_P
        if animation_mode == 'None':
            self.augs = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomAffine(degrees=10,
                               translate=(0.05, 0.05),
                               interpolation=T.InterpolationMode.BILINEAR),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomGrayscale(p=0.1),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.ColorJitter(brightness=0.1,
                              contrast=0.1,
                              saturation=0.1,
                              hue=0.1),
            ])
        elif animation_mode == 'Video Input':
            self.augs = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomPerspective(distortion_scale=0.4, p=0.7),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomGrayscale(p=0.15),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            ])
        elif animation_mode == '2D' or animation_mode == '3D':
            self.augs = T.Compose([
                T.RandomHorizontalFlip(p=0.4),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomAffine(degrees=10,
                               translate=(0.05, 0.05),
                               interpolation=T.InterpolationMode.BILINEAR),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomGrayscale(p=0.1),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.ColorJitter(brightness=0.1,
                              contrast=0.1,
                              saturation=0.1,
                              hue=0.3),
            ])

    def forward(
            self,
            input,
            skip_augs=False,
            heatmap=None,
            padargs={},
            pad_inner=False,
            fix_size=False
    ):
        cutouts = []
        gray = T.Grayscale(3)
        side_y, side_x = input.shape[2:4]

        max_size = min(side_x, side_y)
        min_size = min(side_x, side_y, self.cut_size)
        output_shape = [1, 3, self.cut_size, self.cut_size]

        pad_input = F.pad(
            input,
            ((side_y - max_size) // 2, (side_y - max_size) // 2,
            (side_x - max_size) // 2, (side_x - max_size) // 2),
            **padargs
        )
        cutout = resize(pad_input, out_shape=output_shape)

        # create a list of 1 to 4 in random order, then do the matching overview cut
        # This way even with less than 4 overview cuts, you still get a mix of all of them
        if 0 < self.overview_cut_count <= 4:
            li = [1, 1, 2, 3, 4] # give a slight edge to the normal, full color cut
            ri = [1, 2, 3, 4] #random.sample(li, self.overview_cut_count)
            ri.sort()
            for i in range(self.overview_cut_count):
                if ri[i] == 1:
                    cutouts.append(cutout)
                if ri[i] == 2:
                    cutouts.append(gray(cutout))
                if ri[i] == 3:
                    cutouts.append(TF.hflip(cutout))
                if ri[i] == 4:
                    cutouts.append(gray(TF.hflip(cutout)))
        elif self.overview_cut_count > 4:
            cutout = resize(pad_input, out_shape=output_shape)
            for _ in range(self.overview_cut_count):
                if not skip_augs:
                    cutouts.append(self.augs(cutout))
                else:
                    cutouts.append(cutout)

        innercut_bound_list = []
        for i in range(self.inner_cut_count):
            if fix_size:
                size = min_size
            else:
                size = int(
                    torch.rand([]) ** self.inner_cut_size_exponent * (max_size - min_size) + min_size
                )
            pad_size = int(size / 2) if pad_inner else 0
            if heatmap:
                center_x, center_y = heatmap.sample_centerpoint(size, padded=pad_inner)
                heatmap.add_cut(center_x, center_y, size)
            else:
                inner_mask_size = 0 if pad_inner else int(size / 2)
                center_x, center_y = random_sample(side_x, side_y, inner_mask_size=inner_mask_size)
            if pad_inner:
                pad = T.Pad(pad_size)
                padded = pad(input)
                left, right, top, bottom = center_to_bounds(
                    center_x + pad_size,
                    center_y + pad_size,
                    size,
                    side_x + size,
                    side_y + size
                )
                innercut_bound_list.append((left - pad_size, right - pad_size, top - pad_size, bottom - pad_size))
                cutout = padded[:, :, top:bottom, left:right]
            else:
                left, right, top, bottom = center_to_bounds(center_x, center_y, size, side_x, side_y)
                innercut_bound_list.append((left, right, top, bottom))
                cutout = input[:, :, top:bottom, left:right]
            if i <= int(self.inner_cut_grey_exponent * self.inner_cut_count):
                cutout = gray(cutout)
            cutout = resize(cutout, out_shape=output_shape)
            if not skip_augs:
                cutouts.append(self.augs(cutout))
            else:
                cutouts.append(cutout)
        cutouts = torch.cat(cutouts)
        return cutouts, innercut_bound_list
