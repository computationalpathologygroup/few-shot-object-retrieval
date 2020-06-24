import os
from pprint import pprint

from argparse import ArgumentParser
from encoding import create_encoding
from utils import is_valid_file




# number of cpus
cpus = 6

# parse arguments
parser = ArgumentParser(description='HookNet')
parser.add_argument("-i", '--image_path', dest="image_path", required=True,
                    help="input image path", metavar="FILE_PATH", type=lambda x: is_valid_file(parser, x))
parser.add_argument("-i", '--output_path', dest="output_path", required=True,
                    help="output image path", metavar="FILE_PATH", type=lambda x: is_valid_file(parser, x))
parser.add_argument("-m", '--mask_path', dest="mask_path", required=False,
                    help="mask image path", metavar="FILE_PATH", type=lambda x: is_valid_file(parser, x))
parser.add_argument("-w", '--model_path', dest="model_path", required=True,
                    help="model file path", metavar="FILE_PATH", type=lambda x: is_valid_file(parser, x))
parser.add_argument("-d", '--work_dir', dest="work_dir", required=True,
                    help="work directory", metavar="FILE_PATH", type=lambda x: is_valid_file(parser, x))

config = vars(parser.parse_args())
pprint(f'CONFIG: \n{config}')

create_encoding(model_path=config['model_path'],
                image_path=config['image_path'],
                bg_mask_path=config['mask_path'],
                output_path=config['output_path'],
                cpus=cpus)
