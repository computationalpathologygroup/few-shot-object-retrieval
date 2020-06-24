
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


support = Support(datasource=config['datasource'], model_path=config['model_path'], labels=config['labels'])
prototype, threshold, anchors = support.prototype()

wsienc = WsiEncoding(None)
wsienc.load(config['query_encoding_path'])

enckeys = wsienc._encoding['vectors'].keys()
encvectorkeys = [e for e in wsienc._encoding['vectors'].keys()]

valid_anchor_indexes = []
for idx, enckey in enumerate(wsienc._encoding['vectors'].keys()):
    x, y, w, h, s = enckey
    a = (x+128, y, w, h, s) in wsienc._encoding['vectors']
    b = (x, y+128, w, h, s) in wsienc._encoding['vectors']
    c = (x+128, y+128, w, h, s) in wsienc._encoding['vectors']
    if a and b and c:
        valid_anchor_indexes.append(idx)

accepted_anchors = []
for valid_anchor_index in valid_anchor_indexes:
    x, y, w, h, s = encvectorkeys[valid_anchor_index]
    proposal_patches = []
    proposal_patches.append(wsienc._encoding['vectors'][(x, y, w, h, s )])
    proposal_patches.append(wsienc._encoding['vectors'][(x+128, y, w, h, s )])
    proposal_patches.append(wsienc._encoding['vectors'][(x, y+128, w, h, s )])
    proposal_patches.append(wsienc._encoding['vectors'][(x+128, y+128, w, h, s )])
    sim_value = np.linalg.norm(np.array(proposal_patches).mean(axis=0)-prototype)
    if sim_value <= threshold*1.3:
        accepted_anchors.append(valid_anchor_index)

roi_detections = []
for accepted_anchor in accepted_anchors:
    x,y,w,h,s = encvectorkeys[accepted_anchor]
    roi_detections.append([[x, y], [x, y+256], [x+256, y+256], [x+256, y]])


write_xml(os.path.join(config['output_path'], 'out.xml'), rois=roi_detections)