

import skimage
from keras.models import load_model
import numpy as np

from xmlpathology.xio.imageannotation import ImageAnnotation
from xmlpathology.xio.imagereader import ImageReader, get_image_shape

from utils import to_fully_conv, quantize


class Support:
    def __init__(self, datasource, model_path, labels, grid_cell_size=64, spacing=0.5):
        self._image_annotations = []
        for source in datasource:
            self._image_annotations.append(ImageAnnotation(-1, 
                                                           image_path=source['image_path'], 
                                                           annotation_path=source['annotation_path'],
                                                           label_map={label:idx for idx, label in enumerate(labels)}))
        self._datasource = datasource

        self._model_path = model_path
        self._grid_cell_size = grid_cell_size
        self._spacing = spacing

    def prototype(self):
        model = to_fully_conv(load_model(self._model_path))
        embedding_size = int(model.output.shape[-1])
        embeddings = []
        anchors = set()
        # fig, axes = plt.subplots(1,5,figsize=(10,10))
        axesi =0
        for image_annotation in self._image_annotations:
            support_image = ImageReader(image_annotation.image_path)
            for annotation in image_annotation.annotations:
                ratio = support_image.info[self._spacing]['ratio']
                _, _, width, height = annotation.bounds[0], annotation.bounds[1], annotation.bounds[2]-annotation.bounds[0], annotation.bounds[3]-annotation.bounds[1]
                width, height = quantize(width//ratio, height//ratio, self._grid_cell_size)                
                x,y = annotation.center
                anchors.add((width//64, height//64))
                support_patch=support_image.read_center(x,y,width, height, 0.5)
                # axes[axesi].imshow(support_patch)
                blocks = skimage.util.view_as_blocks(support_patch, (64,64,3)).squeeze().reshape(-1,64,64,3)
                embeddings.append(model.predict_on_batch(blocks/255.0).squeeze().reshape(-1, embedding_size).mean(axis=(0)))  
                # axes[axesi].axis('off')
                axesi +=1
        
        # apply data-augmentations
        
        # find threshold
        thresholds = []
        for findex in range(len(embeddings)):
            for sindex in range(len(embeddings[findex:])):
                thresholds.append(np.linalg.norm(embeddings[findex]-embeddings[sindex]))
                        
        return np.array(embeddings).mean(axis=0), np.max(thresholds), anchors