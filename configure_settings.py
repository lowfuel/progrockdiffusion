import torch
import ast
import numpy as np


def clamp(min_val, val, max_val):
    return max(min_val, min(val, max_val))


def configure_eta(settings, eta):
    default = eta
    if eta == 'auto':
        steps = settings.steps
        maxetasteps = settings.maxetasteps or 315
        minetasteps = settings.minetasteps or 50
        maxeta = settings.maxeta or 1.0
        mineta = settings.mineta - 1.0
        if steps > maxetasteps:
            eta = maxeta
        elif steps < minetasteps:
            eta = mineta
        else:
            stepsrange = (maxetasteps - minetasteps)
            newrange = (maxeta - mineta)
            eta = (((settings.steps - minetasteps)
                    * newrange) / stepsrange) + mineta
            eta = round(eta, 2)
    if eta != default:
        print("eta adjusted to: ", eta)
    return eta


def configure_width_height(settings, width_height):
    # TODO: throw if width or height are incorrect (not existing, negative, too small, etc)
    if(width_height is None):
        width_height = [settings.width, settings.height]
    if isinstance(width_height, str):
        width_height = ast.literal_eval(width_height)
    print(width_height)
    height = (width_height[0] // 64) * 64
    width = (width_height[1] // 64) * 64
    if [height, width] != width_height:
        print(
            f'Changing output size to {width}x{height}. Dimensions must by multiples of 64.'
        )
    return (height, width)


def configure_prompt(settings, full_prompt):
    def parse_prompt(prompt):
        split_prompt = prompt.split(':')
        try:
            weight = float(split_prompt[1])
        except:
            weight = 1.0
        return split_prompt[0], weight
    return [parse_prompt(p) for p in full_prompt.split('|')]


def configure_clip_guidance_scale(settings, clip_guidance_scale):
    if clip_guidance_scale == 'auto':
        w, h = configure_width_height(settings, settings.width_height)
        res = w*h  # total pixels
        maxcgsres = 2000000
        mincgsres = 250000
        maxcgs = 50000
        mincgs = 2500
        if res > maxcgsres:
            clip_guidance_scale = maxcgs
        elif res < mincgsres:
            clip_guidance_scale = mincgs
        else:
            resrange = (maxcgsres - mincgsres)
            newrange = (maxcgs - mincgs)
            clip_guidance_scale = ((
                (res - mincgsres) * newrange) / resrange) + mincgs
            clip_guidance_scale = round(clip_guidance_scale)
        print(f'clip_guidance_scale set automatically to: {clip_guidance_scale}')

    return clamp(1500, clip_guidance_scale, 100000)


def configure_clamp_max(settings, clamp_max):
    if clamp_max == 'auto':
        clamp_max = (settings.steps/1000)**2
    clamp_max = max(.01, min(clamp_max, 1))
    print("clamp_max:", clamp_max)
    return clamp_max


# TODO: handle each schedule
def configure_schedule(settings, schedule):
    if isinstance(schedule, str):
        # TODO: make this safe
        schedule = eval(schedule)
    # TODO: handle non-numbers
    if not isinstance(schedule, list):
        schedule = [schedule]
    return schedule


def configure_device(settings, device):
    if not device:
        device = 'cuda:0'
    if isinstance(device, str):
        device = torch.device(device)
    return device


def configure_symm_switch(settings, symm_switch):
    return clamp(1, symm_switch, settings.steps)


def configure_symmetry_loss_v(settings, symmetry_loss_v):
    symmetry_loss = settings.get('symmetry_loss')
    if symmetry_loss_v is None and symmetry_loss is not None:
        symmetry_loss_v = symmetry_loss
        print("symmetry_loss was depracated, please use symmetry_loss_v in the future")
    return symmetry_loss_v


def configure_width_height(settings, width_height):
    return np.array(width_height) * (settings.get("width_height_scale") or 1)


configurers = {
    "eta": configure_eta,  # TODO: Talk to progrock about this
    "width_height": configure_width_height,
    "prompts": configure_prompt,
    "clip_guidance_scale": configure_clip_guidance_scale,
    "clamp_max": configure_clamp_max,  # TODO: Talk to progrock about this
    "oc_schedule": configure_schedule,
    "ic_schedule": configure_schedule,
    "ic_gray": configure_schedule,
    "ic_bias": configure_schedule,
    "tv_scale": lambda _, v: clamp(0, v, 1000),
    "range_scale": lambda _, v: clamp(0, v, 1000),
    "sat_scale": lambda _, v: clamp(0, v, 20000),
    "clamp_max": lambda _, v: clamp(0.001, v, 0.3),
    "rand_mag": lambda _, v: clamp(0.0, v, 0.999),
    "eta": lambda _, v: clamp(0.0, v, 0.999),
    "cut_ic_pow": lambda _, v: clamp(0.0, v, 100),
    "cut_ic_pow_final": lambda _, v: clamp(0.5, v, 100),
    "symmetry_loss_v": configure_symmetry_loss_v,
    "symm_switch": configure_symm_switch,
    "width_height": configure_width_height
}


def get_configured_settings(settings, configurers=configurers):
    for k, v in settings.items():
        if isinstance(v, str) and v.isnumeric():
            v = float(v)
        if k in configurers:
            settings[k] = configurers[k](settings, v)
    return settings
