import logging
import os

import numexpr
import torch
import clip
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.transforms import functional as transforms_functional
from torch.nn import functional as F

from helpers.vram_helpers import track_model_vram
from cut_modules.make_cutouts import CutHeatmap, save_cut_image, save_inner_cut_bounds_image
from helpers.utils import fetch

logger = logging.getLogger(__name__)

CLIP_NAME_MAP = {
    'ViTB32': 'ViT-B/32',
    'ViTB16': 'ViT-B/16',
    'ViTL14': 'ViT-L/14',
    'ViTL14_336': 'ViT-L/14@336px',
    'RN50': 'RN50',
    'RN50x4': 'RN50x4',
    'RN50x16': 'RN50x16',
    'RN50x64': 'RN50x64',
    'RN101': 'RN101'
}

# These must be norms collected from image rgb channel values?
# What dataset?
clip_img_normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                          std=[0.26862954, 0.26130258, 0.27577711])


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


class ClipManager:

    def __init__(
            self,
            name: str,
            cut_count_multiplier: int,
            device,
            use_cut_heatmap=False,
            pad_inner_cuts=False,
            cutout_debug_image_dir='cutout_debug_images'
    ):
        self.name = name
        self.model = None
        self.cut_count_multiplier = cut_count_multiplier
        self.prompt_weights = None
        self.prompt_embeds = None
        self.device = device
        self.cut_heatmap = None
        self.use_cut_heatmap = use_cut_heatmap
        self.pad_inner_cuts = pad_inner_cuts
        self.cutout_debug_image_dir=cutout_debug_image_dir

    @staticmethod
    def parse_prompt(prompt, vars={}):
        if prompt.startswith('http://') or prompt.startswith('https://'):
            vals = prompt.rsplit(':', 2)
            vals = [vals[0] + ':' + vals[1], *vals[2:]]
        else:
            vals = prompt.rsplit(':', 1)
        vals = vals + ['', '1'][len(vals):]
        return vals[0], float(numexpr.evaluate(vals[1], local_dict=vars))

    def load(self):
        with track_model_vram(self.device, f"Loading {self.name}"):
            self.model = clip.load(
                CLIP_NAME_MAP[self.name],
                jit=False,
                device=self.device
            )[0].eval().requires_grad_(False)

    def embed_text_prompts(
        self,
        prompts,
        step,
        fuzzy_prompt=False,
        fuzzy_prompt_rand_mag=0.05
    ):
        prompt_embeds = []
        prompt_weights = []
        for prompt in prompts:
            txt, weight = self.parse_prompt(prompt, {'s': step})
            encoded_text = self.model.encode_text(clip.tokenize(prompt).to(self.device)).float()
            if fuzzy_prompt:
                for i in range(25):
                    prompt_embeds.append(
                        (
                            encoded_text + torch.randn(encoded_text.shape).to(self.device) * fuzzy_prompt_rand_mag
                        ).clamp(0, 1)
                    )
                    prompt_weights.append(weight)
            else:
                prompt_embeds.append(encoded_text)
                prompt_weights.append(weight)

        prompt_embeds = torch.cat(prompt_embeds)
        prompt_weights = torch.tensor(prompt_weights, device=self.device)
        if prompt_weights.sum().abs() < 1e-3:
            raise RuntimeError('The weights must not sum to 0.')
        return prompt_embeds, prompt_weights

    def embed_image_prompts(
            self,
            prompts,
            step,
            cutn,
            cut_model,
            side_x,
            side_y,
            fuzzy_prompt=False,
            fuzzy_prompt_rand_mag=0.05,
            cutout_skip_augs=False,
            cutout_debug=False
    ):
        cutouts = cut_model(
            self.model.visual.input_resolution,
            cutn,
        )
        prompt_embeds = []
        prompt_weights = []
        for prompt in prompts:
            path, weight = self.parse_prompt(prompt, {'s': step})
            img = Image.open(fetch(path)).convert('RGB')
            img = transforms_functional.resize(
                img,
                min(side_x, side_y, *img.size),
                transforms.InterpolationMode.LANCZOS
            )
            batch, _ = cutouts(
                transforms_functional.to_tensor(img).to(self.device).unsqueeze(0).mul(2).sub(1)
            )
            embed = self.model.encode_image(clip_img_normalize(batch)).float()
            if fuzzy_prompt:
                for i in range(25):
                    prompt_embeds.append(
                        (embed + torch.randn(embed.shape).to(self.device) * fuzzy_prompt_rand_mag).clamp(0, 1))
                    prompt_weights.extend([weight / cutn] * cutn)
            else:
                prompt_embeds.append(embed)
                prompt_weights.extend([weight / cutn] * cutn)

        prompt_embeds = torch.cat(prompt_embeds)
        prompt_weights = torch.tensor(prompt_weights, device=self.device)
        if prompt_weights.sum().abs() < 1e-3:
            raise RuntimeError('The weights must not sum to 0.')
        prompt_weights /= prompt_weights.sum().abs()
        return prompt_embeds, prompt_weights

    def save_debug_images(self, cut_input, innercut_bound_list, cutouts):
        if not os.path.exists(self.cutout_debug_image_dir):
            os.makedirs(self.cutout_debug_image_dir)
        if self.use_cut_heatmap:
            self.cut_heatmap.save_image(os.path.join(self.cutout_debug_image_dir, f"heatmap_{self.name}.jpg"))
        save_inner_cut_bounds_image(
            cut_input,
            innercut_bound_list,
            os.path.join(self.cutout_debug_image_dir, f"inner_cut_bounds_{self.name}.jpg")
        )
        for i, cutout in enumerate(cutouts):
            save_cut_image(cutout, os.path.join(self.cutout_debug_image_dir, f"cutout_{self.name}_{i}.jpg"))

    def get_cut_batch_losses(
            self,
        x_in,
        n,
        cut_overview,
        cut_innercut,
        innercut_power,
        innercut_gray_prob,
        t_int,
        cut_fn,
        cutout_debug=False,
    ):
        try:
            input_resolution = self.model.visual.input_resolution
        except:  # Except what?
            input_resolution = 224

        # Apply model weight to # of overview and innercuts to do
        o_cuts = int(cut_overview[1000 - t_int] * self.cut_count_multiplier)
        i_cuts = int(cut_innercut[1000 - t_int] * self.cut_count_multiplier)
        if o_cuts == 0 and i_cuts == 0:
            i_cuts = 2  # we have to do something otherwise we crash
        logger.debug(f'Doing {o_cuts} overview cuts and {i_cuts} inner for {self.name}')

        if not self.cut_heatmap and self.use_cut_heatmap:
            self.cut_heatmap = CutHeatmap(side_x=x_in.shape[-1], side_y=x_in.shape[-2])

        # Then do the cuts
        cuts = cut_fn(
            input_resolution,
            Overview=o_cuts,
            InnerCrop=i_cuts,
            IC_Size_Pow=innercut_power[1000 - t_int],
            IC_Grey_P=innercut_gray_prob[1000 - t_int]
        )

        cut_input = x_in.add(1).div(2)
        cutouts, innercut_bound_list = cuts(cut_input, heatmap=self.cut_heatmap, pad_inner=self.pad_inner_cuts)
        if cutout_debug:
            self.save_debug_images(cut_input, innercut_bound_list, cutouts)
        if self.use_cut_heatmap:
            self.cut_heatmap.decay()
        clip_in = clip_img_normalize(
            cutouts
        )
        image_embeds = self.model.encode_image(clip_in).float()
        dists = spherical_dist_loss(
            image_embeds.unsqueeze(1),
            self.prompt_embeds.unsqueeze(0))
        dists = dists.view([o_cuts + i_cuts, n, -1])
        losses = dists.mul(self.prompt_weights).sum(2).mean(0)
        return losses
