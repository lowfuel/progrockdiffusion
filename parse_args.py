import argparse


def parse_args():

    python_example = "python3"

    example_text = f'''Usage examples:

    To simply use the 'Default' output directory and get settings from settings.json:
     {python_example} prd.py

    To use your own settings.json (note that putting it in quotes can help parse errors):
     {python_example} prd.py -s "some_directory/mysettings.json"

    Note that multiple settings files are allowed. They're parsed in order. The values present are applied over any previous value:
     {python_example} prd.py -s "some_directory/mysettings.json" -s "highres.json"

    To use the 'Default' output directory and settings, but override the output name and prompt:
     {python_example} prd.py -p "A cool image of the author of this program" -o Coolguy

    To use multiple prompts with optional weight values:
     {python_example} prd.py -p "A cool image of the author of this program" -p "Pale Blue Sky:.5"

    You can ignore the seed coming from a settings file by adding -i, resulting in a new random seed

    To force use of the CPU for image generation, add a -c or --cpu with how many threads to use (warning: VERY slow):
     {python_example} prd.py -c 16

    To generate a checkpoint image at 20% steps, for use as an init image in future runs, add -g or --geninit:
     {python_example} prd.py -g

    To use a checkpoint image at 20% steps add -u or --useinit:
     {python_example} prd.py -u

    To specify which CUDA device to use (advanced) by device ID (default is 0):
     {python_example} prd.py --cuda 1

    To HIDE the settings that get added to your output PNG's metadata, use:
     {python_example} prd.py --hidemetadata

    To increase resolution 2x by splitting the final image and re-rendering detail in the sections, use:
     {python_example} prd.py --gobig

    To increase resolution 2x on an existing output, make sure to supply proper settings, then use:
     {python_example} prd.py --gobig --gobiginit "some_directory/image.png"

    If you already upscaled your gobiginit image, you can skip the resizing process. Provide the scaling factor used:
     {python_example} prd.py --gobig --gobiginit "some_directory/image.png" --gobiginit_scaled 2

    Alternative scaling method is to use ESRGAN (note: RealESRGAN must be installed and in your path):
     {python_example} prd.py --esrgan
    More information on instlaling it is here: https://github.com/xinntao/Real-ESRGAN
    '''

    my_parser = argparse.ArgumentParser(
        prog='ProgRockDiffusion',
        description='Generate images from text prompts.',
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    my_parser.add_argument('--gui',
                           action='store_true',
                           required=False,
                           help='Use the PyQt5 GUI')

    my_parser.add_argument(
        '-s',
        '--settings',
        action='append',
        required=False,
        default=['settings.json'],
        help='A settings JSON file to use, best to put in quotes. Multiples are allowed and layered in order.'
    )

    my_parser.add_argument('-o',
                           '--output',
                           action='store',
                           required=False,
                           help='What output directory to use within images_out')

    my_parser.add_argument('-p',
                           '--prompt',
                           action='append',
                           required=False,
                           help='Override the prompt')

    my_parser.add_argument('-i',
                           '--ignoreseed',
                           action='store_true',
                           required=False,
                           help='Ignores the random seed in the settings file')

    my_parser.add_argument(
        '-c',
        '--cpu',
        type=int,
        nargs='?',
        action='store',
        required=False,
        default=False,
        const=0,
        help='Force use of CPU instead of GPU, and how many threads to run')

    my_parser.add_argument(
        '-g',
        '--geninit',
        type=int,
        nargs='?',
        action='store',
        required=False,
        default=False,
        const=20,
        help='Save a partial image at the specified percent of steps (1 to 99), for use as later init image'
    )
    my_parser.add_argument('-u',
                           '--useinit',
                           action='store_true',
                           required=False,
                           default=False,
                           help='Use the specified init image')

    my_parser.add_argument('--cuda',
                           action='store',
                           required=False,
                           default='0',
                           help='Which GPU to use. Default is 0.')

    my_parser.add_argument(
        '--hidemetadata',
        action='store_true',
        required=False,
        help='Will prevent settings from being added to the output PNG file')

    my_parser.add_argument(
        '--gobig',
        action='store_true',
        required=False,
        help='After generation, the image is split into sections and re-rendered, to double the size.')

    my_parser.add_argument(
        '--gobiginit',
        action='store',
        required=False,
        help='An image to use to kick off GO BIG mode, skipping the initial render.'
    )

    my_parser.add_argument(
        '--gobiginit_scaled',
        type=int,
        nargs='?',
        action='store',
        required=False,
        default=False,
        const=2,
        help='If you already scaled your gobiginit image, add this option along with the multiplier used (default 2)'
    )

    my_parser.add_argument(
        '--esrgan',
        action='store_true',
        required=False,
        help='Resize your output with ESRGAN (realesrgan-ncnn-vulkan must be in your path).'
    )

    my_parser.add_argument(
        '--skip_checks',
        action='store_true',
        required=False,
        default=False,
        help='Do not check values to make sure they are sensible.'
    )

    my_parser.add_argument(
        '--log_level',
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Specify the log level. (default: 'INFO')"
    )

    my_parser.add_argument(
        '--cut_debug',
        action="store_true",
        help="Output cut debug images."
    )

    return my_parser.parse_args()


cl_args = parse_args()
