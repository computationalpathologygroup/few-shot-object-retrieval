import os
from pprint import pprint

from argconfigparser import ArgumentConfigParser
from encoding import create_encoding
from utils import is_valid_file
import glob

parser = ArgumentConfigParser('./encodingparameters.yml', description='FSOR')
config = parser.parse_args()

pprint('CONFIG:')
pprint(config)

# get all masks
masks = []
for mask_path in glob.glob(os.path.join(config['mask_path'], '*.tif')):
    masks.append(mask_path)

# run 
for image_path in glob.glob(os.path.join(config['image_path'], '*.tif')):
    mask_path = None
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    paired_masks = [mask for mask in masks if image_name in mask]
    if len(paired_masks) > 0:
        mask_path = paired_masks[0]

    print(f'encoding image_path: {image_path}')
    print(f'mask_path: {mask_path}')

    # create encoding for image
    create_encoding(model_path=config['model_path'],
                    image_path=image_path,
                    bg_mask_path=mask_path,
                    output_path=config['output_path'],
                    cpus=config['cpus'])
    print(f'done encoding image_path: {image_path}')

print('done encoding images')