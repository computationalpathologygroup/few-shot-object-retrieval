import numpy as np
import os
from pprint import pprint

from support import Support
from encoding import WsiEncoding

from utils import write_xml
from argconfigparser import ArgumentConfigParser
from utils import is_valid_file


# parse arguments
parser = ArgumentConfigParser('./fsorparameters.yml', description='FSOR')
parser.add_argument("-i", '--query_encoding_path', dest="query_encoding_path", required=True,
                    help="query_encoding_path", metavar="FILE_PATH", type=lambda x: is_valid_file(parser, x))
config = parser.parse_args()
pprint(f'CONFIG: \n{config}')

# create support
support = Support(datasource=config['datasource'], model_path=config['model_path'], labels=config['labels'])

# get prototype, threshold and anchors
prototype, threshold, anchors = support.prototype()

# load wsi encoding
wsienc = WsiEncoding(None)
wsienc.load(config['query_encoding_path'])

# get encoding coordinates 
enckeys = wsienc._encoding['vectors'].keys()
encvectorkeys = [e for e in wsienc._encoding['vectors'].keys()]

# get valid anchors
valid_anchor_indexes = []
for idx, enckey in enumerate(wsienc._encoding['vectors'].keys()):
    ratio = 2 #TODO remove hard coding
    true_path_shape = config['patch_shape']*ratio
    x, y, w, h, s = enckey
    a = (x+true_path_shape, y, w, h, s) in wsienc._encoding['vectors']
    b = (x, y+true_path_shape, w, h, s) in wsienc._encoding['vectors']
    c = (x+true_path_shape, y+true_path_shape, w, h, s) in wsienc._encoding['vectors']
    if a and b and c:
        valid_anchor_indexes.append(idx)

# get accepted anchors
accepted_anchors = []
for valid_anchor_index in valid_anchor_indexes:
    x, y, w, h, s = encvectorkeys[valid_anchor_index]
    ratio = 2 #TODO remove hard coding
    true_path_shape = config['patch_shape']*ratio

    # get proposal 
    proposal_patches = []
    proposal_patches.append(wsienc._encoding['vectors'][(x, y, w, h, s )])
    proposal_patches.append(wsienc._encoding['vectors'][(x+true_path_shape, y, w, h, s )])
    proposal_patches.append(wsienc._encoding['vectors'][(x, y+true_path_shape, w, h, s )])
    proposal_patches.append(wsienc._encoding['vectors'][(x+true_path_shape, y+true_path_shape, w, h, s )])

    # check if anchor/proposal is similar to the prototype
    sim_value = np.linalg.norm(np.array(proposal_patches).mean(axis=0)-prototype)
    if sim_value <= threshold*config['threshold_sensitivity']:
        accepted_anchors.append(valid_anchor_index)

# set detections
roi_detections = []
for accepted_anchor in accepted_anchors:
    x,y,w,h,s = encvectorkeys[accepted_anchor]
    ratio = 2 #TODO remove hard coding
    true_path_shape = config['patch_shape']*ratio
    roi_detections.append([[x, y], [x, y+true_path_shape*2], [x+true_path_shape*2, y+true_path_shape*2], [x+true_path_shape*2, y]])

# write detections
write_xml(os.path.join(config['output_path'], 'out.xml'), rois=roi_detections)