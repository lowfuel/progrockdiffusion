import logging
import os

import torch
import pytest

import numpy as np
from cut_modules import make_cutouts
from cut_modules.make_cutouts import (
    MakeCutoutsDango,
    CutHeatmap,
    save_inner_cut_bounds_image,
    save_cut_image,
    random_sample,
    center_to_bounds
)

numeric_log_level = numeric_level = getattr(logging, 'DEBUG', None)
logging.basicConfig(level=numeric_level)
logger = logging.getLogger(__name__)


class TestHeatmapSampling:

    def test_heatmap_sample_padded(self):
        image_x, image_y = 125, 100
        # This test is probabilistic, which isn't ideal, but should be adequate here.
        # There's a very small chance of a false positive, but no chance of false negative.
        heatmap = CutHeatmap(side_x=image_x, side_y=image_y)
        cut_size = 50

        y_list = []
        x_list = []

        for _ in range(1000):
            x, y = heatmap.sample_centerpoint(cut_size, padded=True)
            y_list.append(y)
            x_list.append(x)

        assert max(x_list) < image_x
        assert max(y_list) < image_y

    def test_heatmap_sample_unpadded(self):
        image_x, image_y = 125, 100
        # This test is probabilistic, which isn't ideal, but should be adequate here.
        # There's a very small chance of a false positive, but no chance of false negative.
        heatmap = CutHeatmap(side_x=image_x, side_y=image_y)
        cut_size = 50
        pad_size = int(50 / 2) - 1

        y_list = []
        x_list = []

        for _ in range(1000):
            x, y = heatmap.sample_centerpoint(cut_size, padded=False)
            y_list.append(y)
            x_list.append(x)

        assert max(x_list) < image_x - pad_size
        assert max(y_list) < image_y - pad_size
        assert min(x_list) >= pad_size
        assert min(y_list) >= pad_size


class TestRandomSampling:
    def test_random_sample_padded(self):
        """
        If we're adding padding to the outside of the image to allow cuts to extend
        over the edge of the image, we want our center point samples to range all the
        way to the edge of the image.
        """
        image_x, image_y = 125, 100
        cut_size = 50
        pad_size = int(cut_size / 2)

        y_list = []
        x_list = []

        for _ in range(1000):
            x, y = random_sample(side_x=image_x, side_y=image_y)
            y_list.append(y)
            x_list.append(x)

        # Check that our sampled x and y values are all within the image, but can extend up to the edges
        assert image_x - pad_size < max(x_list) < image_x
        assert image_y - pad_size < max(y_list) < image_y

    def test_random_sample_unpadded(self):
        image_x, image_y = 125, 100
        cut_size = 50
        pad_size = int(cut_size / 2)

        y_list = []
        x_list = []

        for _ in range(1000):
            x, y = random_sample(side_x=image_x, side_y=image_y, inner_mask_size=pad_size)
            y_list.append(y)
            x_list.append(x)

        # Check that our sampled x and y values are all far enough from the edge that the cut won't
        # extend beyond the edge
        assert max(x_list) < image_x - pad_size
        assert max(y_list) < image_y - pad_size


class TestCutouts:

    @staticmethod
    def save_test_images(name_prefix, image, bounds_list, cuts, heatmap):
        test_output_dir = 'test_output'
        if not os.path.exists(test_output_dir):
            os.makedirs(test_output_dir)
        if heatmap:
            heatmap.save_image(os.path.join(test_output_dir, f"{name_prefix}_heatmap.jpg"))
        save_inner_cut_bounds_image(image, bounds_list, (os.path.join(test_output_dir, f"{name_prefix}_bounds.jpg")))
        for i, cut in enumerate(cuts):
            save_cut_image(cut, os.path.join(test_output_dir, f"{name_prefix}_cut_{i}.jpg"))

    @pytest.fixture
    def cutout_module(self):
        return MakeCutoutsDango(
            cut_size=20,
            Overview=0,
            InnerCrop=1,
            IC_Size_Pow=100,
        )

    @pytest.fixture
    def image_and_heatmap(self, request):
        image_x, image_y = request.param
        heatmap = CutHeatmap(side_x=image_x, side_y=image_y)
        test_image = torch.ones((1, 3, image_y, image_x))
        heatmap.heatmap = np.zeros(test_image.shape[-2:], dtype=np.float32)
        for i in range(test_image.shape[-2]):
            if i % 10 == 0:
                test_image[:, :, i - 4:i, :] = 0.5
        for j in range(test_image.shape[-1]):
            if j % 10 == 0:
                test_image[:, :, :, j - 4:j] = 0.5
        return test_image, heatmap

    @pytest.mark.parametrize('image_and_heatmap', [(125, 100)], indirect=True)
    def test_centered_padded_heatmap_cutout(self, image_and_heatmap, cutout_module):
        image, heatmap = image_and_heatmap
        heatmap.heatmap[50, 50] = 1.0
        cuts, bounds_list = cutout_module(
            image,
            pad_inner=True,
            heatmap=heatmap,
            fix_size=True,
            skip_augs=True
        )
        self.save_test_images("test_centered_padded_heatmap_cutout", image, bounds_list, cuts, heatmap)
        assert bounds_list[0] == (40, 60, 40, 60)
        assert cuts[0].shape == (3, 20, 20)

    @pytest.mark.parametrize('image_and_heatmap', [(125, 100)], indirect=True)
    def test_corner_padded_heatmap_cutout(self, image_and_heatmap, cutout_module):
        image, heatmap = image_and_heatmap
        heatmap.heatmap[0, 0] = 1.0
        cuts, bounds_list = cutout_module(
            image,
            pad_inner=True,
            heatmap=heatmap,
            fix_size=True,
            skip_augs=True
        )
        self.save_test_images("test_corner_padded_heatmap_cutout", image, bounds_list, cuts, heatmap)
        assert bounds_list[0] == (-10, 10, -10, 10)
        assert cuts[0].shape == (3, 20, 20)

    @pytest.mark.parametrize('image_and_heatmap', [(125, 100)], indirect=True)
    def test_corner_unpadded_heatmap_cutout(self, image_and_heatmap, cutout_module):
        image, heatmap = image_and_heatmap
        heatmap.heatmap[10, 114] = 1.0
        cuts, bounds_list = cutout_module(
            image,
            pad_inner=False,
            heatmap=heatmap,
            fix_size=True,
            skip_augs=True
        )
        self.save_test_images("test_corner_unpadded_heatmap_cutout", image, bounds_list, cuts, heatmap)
        assert bounds_list[0] == (104, 124, 0, 20)
        assert cuts[0].shape == (3, 20, 20)

    @pytest.mark.parametrize('image_and_heatmap', [(125, 100)], indirect=True)
    def test_corner_unpadded_no_heatmap_cutout(self, image_and_heatmap, cutout_module, monkeypatch):
        def mock_random_sample(*args, **kwargs):
            return 114, 10  # center_x, center_y
        monkeypatch.setattr(make_cutouts, "random_sample", mock_random_sample)
        image, _ = image_and_heatmap
        cuts, bounds_list = cutout_module(
            image,
            pad_inner=False,
            fix_size=True,
            skip_augs=True
        )
        self.save_test_images("test_corner_unpadded_no_heatmap_cutout", image, bounds_list, cuts, None)
        assert bounds_list[0] == (104, 124, 0, 20)
        assert cuts[0].shape == (3, 20, 20)
