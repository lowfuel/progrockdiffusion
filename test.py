# Load the JSON config files
import random
# Simple check to see if a key is present in the settings file
import json5 as json


def is_json_key_present(json, key, subkey="none"):
    try:
        print(key, json[key])
    except:
        print(key, "-----NOT FOUND-----")


# A simple way to ensure values are in an accceptable range, and also return a random value if desired
def clampval(var_name, minval, val, maxval):
    if val == "random":
        try:
            val = random.randint(minval, maxval)
        except:
            val = random.uniform(minval, maxval)
        return val
    # Auto is handled later, so we just return it back as is
    elif val == "auto":
        return val
    elif type(val) == str:
        return val
    elif val < minval:
        print(f'Warning: {var_name} is below {minval} - if you get bad results, consider adjusting.')
        return val
    elif val > maxval:
        print(f'Warning: {var_name} is above {maxval} - if you get bad results, consider adjusting.')
        return val
    else:
        return val


with open("default_settings.json", 'r', encoding="utf-8") as json_file:
    settings_file = json.load(json_file)
    # If any of these are in this settings file they'll be applied, overwriting any previous value.
    # Some are passed through clampval first to make sure they are within bounds (or randomized if desired)
    if is_json_key_present(settings_file, 'batch_name'):
        batch_name = (settings_file['batch_name'])
    if is_json_key_present(settings_file, 'text_prompts'):
        text_prompts = (settings_file['text_prompts'])
    if is_json_key_present(settings_file, 'image_prompts'):
        image_prompts = (settings_file['image_prompts'])
    if is_json_key_present(settings_file, 'clip_guidance_scale'):
        clip_guidance_scale = clampval('clip_guidance_scale',
                                       1500, (settings_file['clip_guidance_scale']), 100000)
    if is_json_key_present(settings_file, 'tv_scale'):
        tv_scale = clampval('tv_scale', 0, (settings_file['tv_scale']), 1000)
    if is_json_key_present(settings_file, 'range_scale'):
        range_scale = clampval('range_scale', 0, (settings_file['range_scale']), 1000)
    if is_json_key_present(settings_file, 'sat_scale'):
        sat_scale = clampval('sat_scale', 0, (settings_file['sat_scale']), 20000)
    if is_json_key_present(settings_file, 'n_batches'):
        n_batches = (settings_file['n_batches'])
    if is_json_key_present(settings_file, 'display_rate'):
        display_rate = (settings_file['display_rate'])
    if is_json_key_present(settings_file, 'cutn_batches'):
        cutn_batches = (settings_file['cutn_batches'])
    if is_json_key_present(settings_file, 'cutn_batches_final'):
        cutn_batches_final = (settings_file['cutn_batches_final'])
    if is_json_key_present(settings_file, 'max_frames'):
        max_frames = (settings_file['max_frames'])
    if is_json_key_present(settings_file, 'interp_spline'):
        interp_spline = (settings_file['interp_spline'])
    if is_json_key_present(settings_file, 'init_image'):
        init_image = (settings_file['init_image'])
    if is_json_key_present(settings_file, 'init_scale'):
        init_scale = (settings_file['init_scale'])
    if is_json_key_present(settings_file, 'skip_steps'):
        skip_steps = (settings_file['skip_steps'])
    if is_json_key_present(settings_file, 'skip_steps_ratio'):
        skip_steps_ratio = (settings_file['skip_steps_ratio'])
    if is_json_key_present(settings_file, 'stop_early'):
        stop_early = (settings_file['stop_early'])
    if is_json_key_present(settings_file, 'frames_scale'):
        frames_scale = (settings_file['frames_scale'])
    if is_json_key_present(settings_file, 'frames_skip_steps'):
        frames_skip_steps = (settings_file['frames_skip_steps'])
    if is_json_key_present(settings_file, 'perlin_init'):
        perlin_init = (settings_file['perlin_init'])
    if is_json_key_present(settings_file, 'perlin_mode'):
        perlin_mode = (settings_file['perlin_mode'])
    if is_json_key_present(settings_file, 'skip_augs'):
        skip_augs = (settings_file['skip_augs'])
    if is_json_key_present(settings_file, 'randomize_class'):
        randomize_class = (settings_file['randomize_class'])
    if is_json_key_present(settings_file, 'clip_denoised'):
        clip_denoised = (settings_file['clip_denoised'])
    if is_json_key_present(settings_file, 'clamp_grad'):
        clamp_grad = (settings_file['clamp_grad'])
    if is_json_key_present(settings_file, 'clamp_max'):
        clamp_max = clampval('clamp_max', 0.001, (settings_file['clamp_max']), 0.3)
    if is_json_key_present(settings_file, 'set_seed'):
        set_seed = (settings_file['set_seed'])
    if is_json_key_present(settings_file, 'fuzzy_prompt'):
        fuzzy_prompt = (settings_file['fuzzy_prompt'])
    if is_json_key_present(settings_file, 'rand_mag'):
        rand_mag = clampval('rand_mag', 0.0, (settings_file['rand_mag']), 0.999)
    if is_json_key_present(settings_file, 'eta'):
        eta = clampval('eta', 0.0, (settings_file['eta']), 0.999)
    if is_json_key_present(settings_file, 'width'):
        width_height = [(settings_file['width']),
                        (settings_file['height'])]
    if is_json_key_present(settings_file, 'width_height_scale'):
        width_height_scale = (settings_file['width_height_scale'])
    if is_json_key_present(settings_file, 'diffusion_model'):
        diffusion_model = (settings_file['diffusion_model'])
    if is_json_key_present(settings_file, 'use_secondary_model'):
        use_secondary_model = (settings_file['use_secondary_model'])
    if is_json_key_present(settings_file, 'steps'):
        steps = (settings_file['steps'])
    if is_json_key_present(settings_file, 'sampling_mode'):
        sampling_mode = (settings_file['sampling_mode'])
    if is_json_key_present(settings_file, 'diffusion_steps'):
        diffusion_steps = (settings_file['diffusion_steps'])
    if is_json_key_present(settings_file, 'ViTB32'):
        ViTB32 = (settings_file['ViTB32'])
    if is_json_key_present(settings_file, 'ViTB16'):
        ViTB16 = (settings_file['ViTB16'])
    if is_json_key_present(settings_file, 'ViTL14'):
        ViTL14 = (settings_file['ViTL14'])
    if is_json_key_present(settings_file, 'ViTL14_336'):
        ViTL14_336 = (settings_file['ViTL14_336'])
    if is_json_key_present(settings_file, 'RN101'):
        RN101 = (settings_file['RN101'])
    if is_json_key_present(settings_file, 'RN50'):
        RN50 = (settings_file['RN50'])
    if is_json_key_present(settings_file, 'RN50x4'):
        RN50x4 = (settings_file['RN50x4'])
    if is_json_key_present(settings_file, 'RN50x16'):
        RN50x16 = (settings_file['RN50x16'])
    if is_json_key_present(settings_file, 'RN50x64'):
        RN50x64 = (settings_file['RN50x64'])
    if is_json_key_present(settings_file, 'cut_overview'):
        cut_overview = (settings_file['cut_overview'])
    if is_json_key_present(settings_file, 'cut_innercut'):
        cut_innercut = (settings_file['cut_innercut'])
    if is_json_key_present(settings_file, 'cut_ic_pow'):
        cut_ic_pow = (settings_file['cut_ic_pow'])
        if type(cut_ic_pow) != str:
            cut_ic_pow = clampval('cut_ic_pow', 0.0, cut_ic_pow, 100)
    if is_json_key_present(settings_file, 'cut_ic_pow_final'):
        cut_ic_pow_final = clampval('cut_ic_pow_final', 0.5, (settings_file['cut_ic_pow_final']), 100)
    if is_json_key_present(settings_file, 'cut_icgray_p'):
        cut_icgray_p = (settings_file['cut_icgray_p'])
    if is_json_key_present(settings_file, 'smooth_schedules'):
        smooth_schedules = (settings_file['smooth_schedules'])
    if is_json_key_present(settings_file, 'key_frames'):
        key_frames = (settings_file['key_frames'])
    if is_json_key_present(settings_file, 'angle'):
        angle = (settings_file['angle'])
    if is_json_key_present(settings_file, 'zoom'):
        zoom = (settings_file['zoom'])
    if is_json_key_present(settings_file, 'translation_x'):
        translation_x = (settings_file['translation_x'])
    if is_json_key_present(settings_file, 'translation_y'):
        translation_y = (settings_file['translation_y'])
    if is_json_key_present(settings_file, 'video_init_path'):
        video_init_path = (settings_file['video_init_path'])
    if is_json_key_present(settings_file, 'extract_nth_frame'):
        extract_nth_frame = (settings_file['extract_nth_frame'])
    if is_json_key_present(settings_file, 'intermediate_saves'):
        intermediate_saves = (settings_file['intermediate_saves'])
    if is_json_key_present(settings_file, 'fix_brightness_contrast'):
        fix_brightness_contrast = (settings_file['fix_brightness_contrast'])
    if is_json_key_present(settings_file, 'adjustment_interval'):
        adjustment_interval = (settings_file['adjustment_interval'])
    if is_json_key_present(settings_file, 'high_contrast_threshold'):
        high_contrast_threshold = (
            settings_file['high_contrast_threshold'])
    if is_json_key_present(settings_file,
                           'high_contrast_adjust_amount'):
        high_contrast_adjust_amount = (
            settings_file['high_contrast_adjust_amount'])
    if is_json_key_present(settings_file, 'high_contrast_start'):
        high_contrast_start = (settings_file['high_contrast_start'])
    if is_json_key_present(settings_file, 'high_contrast_adjust'):
        high_contrast_adjust = (settings_file['high_contrast_adjust'])
    if is_json_key_present(settings_file, 'low_contrast_threshold'):
        low_contrast_threshold = (
            settings_file['low_contrast_threshold'])
    if is_json_key_present(settings_file,
                           'low_contrast_adjust_amount'):
        low_contrast_adjust_amount = (
            settings_file['low_contrast_adjust_amount'])
    if is_json_key_present(settings_file, 'low_contrast_start'):
        low_contrast_start = (settings_file['low_contrast_start'])
    if is_json_key_present(settings_file, 'low_contrast_adjust'):
        low_contrast_adjust = (settings_file['low_contrast_adjust'])
    if is_json_key_present(settings_file, 'high_brightness_threshold'):
        high_brightness_threshold = (
            settings_file['high_brightness_threshold'])
    if is_json_key_present(settings_file,
                           'high_brightness_adjust_amount'):
        high_brightness_adjust_amount = (
            settings_file['high_brightness_adjust_amount'])
    if is_json_key_present(settings_file, 'high_brightness_start'):
        high_brightness_start = (
            settings_file['high_brightness_start'])
    if is_json_key_present(settings_file, 'high_brightness_adjust'):
        high_brightness_adjust = (
            settings_file['high_brightness_adjust'])
    if is_json_key_present(settings_file, 'low_brightness_threshold'):
        low_brightness_threshold = (
            settings_file['low_brightness_threshold'])
    if is_json_key_present(settings_file,
                           'low_brightness_adjust_amount'):
        low_brightness_adjust_amount = (
            settings_file['low_brightness_adjust_amount'])
    if is_json_key_present(settings_file, 'low_brightness_start'):
        low_brightness_start = (settings_file['low_brightness_start'])
    if is_json_key_present(settings_file, 'low_brightness_adjust'):
        low_brightness_adjust = (
            settings_file['low_brightness_adjust'])
    if is_json_key_present(settings_file, 'sharpen_preset'):
        sharpen_preset = (settings_file['sharpen_preset'])
    if is_json_key_present(settings_file, 'keep_unsharp'):
        keep_unsharp = (settings_file['keep_unsharp'])
    if is_json_key_present(settings_file, 'animation_mode'):
        animation_mode = (settings_file['animation_mode'])
    if is_json_key_present(settings_file, 'gobig_orientation'):
        gobig_orientation = (settings_file['gobig_orientation'])
    if is_json_key_present(settings_file, 'gobig_scale'):
        gobig_scale = int(settings_file['gobig_scale'])
    if is_json_key_present(settings_file, 'symmetry_loss'):
        symmetry_loss_v = (settings_file['symmetry_loss'])
        print("symmetry_loss was depracated, please use symmetry_loss_v in the future")
    if is_json_key_present(settings_file, 'symmetry_loss_v'):
        symmetry_loss_v = (settings_file['symmetry_loss_v'])
    if is_json_key_present(settings_file, 'symmetry_loss_h'):
        symmetry_loss_h = (settings_file['symmetry_loss_h'])
    if is_json_key_present(settings_file, 'symm_loss_scale'):
        symm_loss_scale = (settings_file['symm_loss_scale'])
    if is_json_key_present(settings_file, 'symm_switch'):
        symm_switch = int(clampval('symm_switch', 1, (settings_file['symm_switch']), steps))
