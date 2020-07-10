

import skimage
from keras.models import load_model
import numpy as np

from xmlpathology.batchgenerator.data.wholeslideimage import WholeSlideImageOpenSlide
from xmlpathology.batchgenerator.data.wholeslideannotation import WholeSlideAnnotation


from utils import to_fully_conv, quantize

from scipy.stats.mstats import gmean

from numpy import dot
from numpy.linalg import norm

import imgaug as ia
from imgaug import augmenters as iaa
def augmentor(images):
        'Apply data augmentation'
        seq = iaa.Sequential(
                [
                # apply the following augmenters to most images
                iaa.Fliplr(0.9),  # horizontally flip 50% of all images
                iaa.Flipud(0.9),  # vertically flip 20% of all images
                iaa.ElasticTransformation(alpha=(10, 20), sigma=6),
                iaa.Multiply((0.7, 1.2), per_channel=0.4),
                iaa.GaussianBlur((0, 0.5)),             
                ],
                random_order=True
        )
        return seq.augment_images(images)


class Support:
    def __init__(self, datasource, model_path, labels, grid_cell_size=64, spacing=0.5, calculate_threshold=np.mean):
        self._image_annotations = []
        for source in datasource:
            self._image_annotations.append(WholeSlideAnnotation(-1, 
                                                               image_path=source['image_path'], 
                                                               annotation_path=source['annotation_path'],
                                                               label_map={label:idx for idx, label in enumerate(labels)}))
        self._datasource = datasource

        self._model_path = model_path
        self._grid_cell_size = grid_cell_size
        self._spacing = spacing
        self._calculate_threshold = calculate_threshold

    def prototype(self):
        model = load_model(self._model_path)
        embedding_size = int(model.output.shape[-1])
        embeddings = []
        anchors = set()
        axesi =0
        patches = []
        for image_annotation in self._image_annotations:
            support_image = WholeSlideImageOpenSlide(image_annotation.image_path)
            for annotation in image_annotation.annotations:
                ratio = support_image.get_downsampling_from_spacing(self._spacing)
                _, _, width, height = annotation.bounds[0], annotation.bounds[1], annotation.bounds[2]-annotation.bounds[0], annotation.bounds[3]-annotation.bounds[1]
                width, height = quantize(width//ratio, height//ratio, self._grid_cell_size)                
                x,y = annotation.center
                anchors.add((width//64, height//64))
                support_patch=support_image.get_patch(x,y,width, height, 0.5)
                patches.append(support_patch)
                blocks = skimage.util.view_as_blocks(support_patch, (64,64,3)).squeeze().reshape(-1,64,64,3)
                # apply data-augmentations
                blocks_augmented = augmentor(blocks)
                all_blocks = np.concatenate([blocks, blocks_augmented])
                embeddings.append(model.predict_on_batch(all_blocks/255.0).squeeze().reshape(-1, embedding_size).mean(axis=(0)))  
        
        proto = np.array(embeddings).mean(axis=0)
        # find threshold
        thresholds = []
        cos_thresholds = []
        for findex in range(len(embeddings)):   
                # cos_thresholds = dot(embeddings[findex], embeddings[sindex])/(norm(embeddings[findex])*norm(embeddings[sindex]))
                thresholds.append(np.linalg.norm(embeddings[findex]-proto))
        
        thresholds = [t for t in thresholds if t != 0]
        return np.array(embeddings).mean(axis=0), self._calculate_threshold(thresholds), anchors, patches, thresholds