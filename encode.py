import os
from pprint import pprint

from argconfigparser import ArgumentConfigParser
from encoding import create_encoding
from utils import is_valid_file

parser = ArgumentConfigParser('./encodingparameters.yml', description='FSOR')
config = parser.parse_args()

pprint('CONFIG:')
pprint(config)

# create encoding for image
create_encoding(model_path=config['model_path'],
                image_path=config['image_path'],
                bg_mask_path=config['mask_path'],
                output_path=config['output_path'],
                cpus=config['cpus'])
