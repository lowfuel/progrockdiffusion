import logging

logger = logging.getLogger(__name__)


def clamp(min_val, val, max_val):
    return max(min_val, min(val, max_val))


def val_interpolate(x1, y1, x2, y2, x):
    # Linear interpolation. Return y between y1 and y2 for the same position x is bettewen x1 and x2
    d = [[x1, y1], [x2, y2]]
    output = d[0][1] + (x - d[0][0]) * ((d[1][1] - d[0][1])/(d[1][0] - d[0][0]))
    if type(y1) == int:
        output = int(output)  # return the proper type
    return(output)


def num_to_schedule(input, final=-9999):
    # take a single number and turn it into a string-style schedule, with support for interpolated
    if final != -9999:
        output = (f"[{input}]*1+")
        for i in range(1, 1000):
            val = val_interpolate(1, input, 1000, final, i)
            output = output + (f"[{val}]*1+")
        output = output[:-1]  # remove the final plus character
    else:
        output = (f'[{input}]*1000')
    return(output)
# Automatic clip_guidance_scale based on overall resolution


def config_clip_guidance_scale(settings, clip_guidance_scale):
    if clip_guidance_scale == 'auto':
        res = settings.width_height[0] * settings.width_height[1]  # total pixels
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
            clip_guidance_scale = (((res - mincgsres) * newrange) / resrange) + mincgs
            clip_guidance_scale = round(clip_guidance_scale)
        clip_guidance_scale = num_to_schedule(clip_guidance_scale)
        print(f'clip_guidance_scale set automatically to: {clip_guidance_scale}')
    return clip_guidance_scale


# Automatic Eta based on steps
def config_eta(settings, eta):
    if settings.eta == 'auto':
        maxetasteps = 315
        minetasteps = 50
        maxeta = 1.0
        mineta = 0.0
        if settings.steps > maxetasteps:
            eta = maxeta
        elif settings.steps < minetasteps:
            eta = mineta
        else:
            stepsrange = (maxetasteps - minetasteps)
            newrange = (maxeta - mineta)
            eta = (((settings.steps - minetasteps) * newrange) / stepsrange) + mineta
            eta = round(eta, 2)
            print(f'Eta set automatically to: {eta}')
    return eta


def config_clamp_max(settings, clamp_max):
    # Automatic clamp_max based on steps
    if clamp_max == 'auto':
        if settings.steps <= 35:
            clamp_max = 0.001
        elif settings.steps <= 75:
            clamp_max = 0.0125
        elif settings.steps <= 150:
            clamp_max = 0.02
        elif settings.steps <= 225:
            clamp_max = 0.035
        elif settings.steps <= 300:
            clamp_max = 0.05
        elif settings.steps <= 500:
            clamp_max = 0.075
        else:
            clamp_max = 0.1
        if settings.use_secondary_model == False:
            clamp_max = clamp_max * 2
        clamp_max = num_to_schedule(clamp_max)
        print(f'Clamp_max automatically set to {clamp_max}')
    elif type(clamp_max) != str:
        clamp_max = num_to_schedule(clamp_max)
        print(f'Converted clamp_max to schedule, new value is: {clamp_max}')
    return clamp_max


def config_cutn_batches(settings, cutn_batches):
    if type(cutn_batches) != str:
        if settings.cutn_batches_final != None:
            cutn_batches = num_to_schedule(cutn_batches, settings.cutn_batches_final)
        else:
            cutn_batches = num_to_schedule(cutn_batches)
        print(f'Converted cutn_batches to schedule.')
        logger.debug(f'cutn_batches schedule is: {cutn_batches}')
    return cutn_batches


configurers = {
    'clip_guidance_scale': config_clip_guidance_scale,
    'eta': config_eta,
    'clamp_max': config_clamp_max,
    'cutn_batches': config_cutn_batches
}
