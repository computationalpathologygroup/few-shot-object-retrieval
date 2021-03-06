import os
import numpy as np

from tqdm import tqdm
from keras.models import load_model

from xmlpathology.batchgenerator.generators import WSIPatchGenerator
from xmlpathology.batchgenerator.data.wholeslideimage import WholeSlideImageASAP, WholeSlideImageOpenSlide




class WsiEncoding:
    def __init__(self, image_path):
        self._encoding = {'image_path': image_path,
                          'vectors': {}}
        
        if image_path:
            image = WholeSlideImageOpenSlide(image_path)
            shape = (np.array(image.shapes[0])+1024)//64
            layout = np.zeros((shape[1], shape[0]))
            self._encoding['layout'] = layout
    
    def load(self, path):
        self._encoding = np.load(path, allow_pickle=True).item()

    def add(self, x, y, width, height, spacing, encoding, overwrite=False):
        if not overwrite and (x, y,width,height,spacing) in self._encoding:
            raise ValueError(f'{(x,y,width,height,spacing)} encoding already present and overwrite={overwrite}')
            
        self._encoding['vectors'][(x, y, width, height, spacing)] = encoding
        self._encoding['layout'][y//64][x//64] = 1
        
    def save(self, output_path):
        np.save(output_path, self._encoding)


def create_encoding(model_path,
                     image_path,
                     output_path,
                     bg_mask_path=None,
                     patch_shape=(64, 64, 3),
                     spacing=0.5,
                     tile_shape=(1024, 1024),
                     shift=(1024, 1024),
                     cpus=1):
    # Init model
    model = load_model(model_path, compile=False)
    image = WholeSlideImageOpenSlide(image_path)
    shape = image.shapes[image.get_level_from_spacing(spacing)]
    
    bg_mask = None
    bg_encoding = None
    if bg_mask_path:
        bg_mask = WholeSlideImageASAP(bg_mask_path)
    
    wsi_encoding = WsiEncoding(image_path)
    wsi_encoding_output_path = os.path.join(output_path,os.path.basename(image_path).replace('.mrxs', '_4task_encoded.npy'))
    
    # set datasources
    data_sources = {'training': [{'image_path': image_path,
                                  'annotation_path': image_path}]}

    # Init batchgenerator
    batchgen = WSIPatchGenerator(data_sources=data_sources,
                                 tile_shape=tile_shape,
                                 patch_shape=patch_shape,
                                 shift=shift,
                                 cpus=cpus-1,
                                 spacing=spacing,
                                 log_path=output_path)

    # start batchgenerator
    batchgen.start()
    
    # loop over data
    for image_annotation in batchgen.datasets['training'].image_annotations:

        # loop over annotations
        for i, annotation in enumerate(tqdm(image_annotation.annotations)):
            
            # get batch
            batch = batchgen.batch('training')
            
            mask_patch = None
            if bg_mask:
                mask_patch = bg_mask.get_patch(annotation.bounds[0]//4, annotation.bounds[1]//4, tile_shape[0]//4, tile_shape[1]//4, 2.0)
            
            if mask_patch is None or np.any(mask_patch):

                # encode patches
                encodings = np.round(model.predict_on_batch(batch), 4)

                # set width heigth of encoded tile
                w = tile_shape[0]//patch_shape[0]
                h = tile_shape[1]//patch_shape[1]

                # reshape encodings to tile
                encodings = encodings.reshape(h, w, -1)

                # set encoding patches
                ratio =1
                for w_index in range(w):
                    for h_index in range(h):
                        x = annotation.bounds[0] + patch_shape[0] * w_index * ratio
                        y = annotation.bounds[1] + patch_shape[1] * h_index * ratio
                        wsi_encoding.add(x, y, patch_shape[0], patch_shape[1], spacing, encodings[h_index][w_index])

    # saving
    wsi_encoding.save(wsi_encoding_output_path)
    batchgen.stop()




    