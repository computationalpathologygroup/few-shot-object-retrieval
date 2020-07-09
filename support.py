

import skimage
from keras.models import load_model
import numpy as np

from xmlpathology.batchgenerator.data.wholeslideimage import WholeSlideImageOpenSlide
from xmlpathology.batchgenerator.data.wholeslideannotation import WholeSlideAnnotation


from utils import to_fully_conv, quantize

import imgaug as ia
from imgaug import augmenters as iaa
def augmentor(images):
        'Apply data augmentation'
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        seq = iaa.Sequential(
                [
                # apply the following augmenters to most images
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Flipud(0.2),  # vertically flip 20% of all images
                sometimes(iaa.Affine(
                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                    # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    # translate by -20 to +20 percent (per axis)
                    rotate=(-10, 10),  # rotate by -45 to +45 degrees
                    shear=(-5, 5),  # shear by -16 to +16 degrees
                    order=[0, 1],
                    # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                    mode=ia.ALL
                    # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                           [sometimes(iaa.Superpixels(p_replace=(0, 1.0),
                                                             n_segments=(20, 200))),
                               # convert images into their superpixel representation
                               iaa.OneOf([
                                       iaa.GaussianBlur((0, 1.0)),
                                       # blur images with a sigma between 0 and 3.0
                                       iaa.AverageBlur(k=(3, 5)),
                                       # blur image using local means with kernel sizes between 2 and 7
                                       iaa.MedianBlur(k=(3, 5)),
                                       # blur image using local medians with kernel sizes between 2 and 7
                               ]),
                               iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)),
                               # sharpen images
                               iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                               # emboss images
                               # search either for all edges or for directed edges,
                               # blend the result with the original image using a blobby mask
                               iaa.SimplexNoiseAlpha(iaa.OneOf([
                                       iaa.EdgeDetect(alpha=(0.5, 1.0)),
                                       iaa.DirectedEdgeDetect(alpha=(0.5, 1.0),
                                                              direction=(0.0, 1.0)),
                               ])),
                               iaa.AdditiveGaussianNoise(loc=0,
                                                         scale=(0.0, 0.01 * 255),
                                                         per_channel=0.5),
                               # add gaussian noise to images
                               iaa.OneOf([
                                       iaa.Dropout((0.01, 0.05), per_channel=0.5),
                                       # randomly remove up to 10% of the pixels
                                       iaa.CoarseDropout((0.01, 0.03),
                                                         size_percent=(0.01, 0.02),
                                                         per_channel=0.2),
                               ]),
                               iaa.Invert(0.01, per_channel=True),
                               # invert color channels
                               iaa.Add((-2, 2), per_channel=0.5),
                               # change brightness of images (by -10 to 10 of original value)
                               iaa.AddToHueAndSaturation((-1, 1)),
                               # change hue and saturation
                               # either change the brightness of the whole image (sometimes
                               # per channel) or change the brightness of subareas
                               iaa.OneOf([
                                       iaa.Multiply((0.9, 1.1), per_channel=0.5),
                                       iaa.FrequencyNoiseAlpha(
                                               exponent=(-1, 0),
                                               first=iaa.Multiply((0.9, 1.1),
                                                                  per_channel=True),
                                               second=iaa.ContrastNormalization(
                                                       (0.9, 1.1))
                                       )
                               ]),
                               sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5),
                                                                   sigma=0.25)),
                               # move pixels locally around (with random strengths)
                               sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                               # sometimes move parts of the image around
                               sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                           ],
                           random_order=True
                           )
                ],
                random_order=True
        )
        return seq.augment_images(images)


class Support:
    def __init__(self, datasource, model_path, labels, grid_cell_size=64, spacing=0.5):
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
        
        # find threshold
        thresholds = []
        for findex in range(len(embeddings)):
            for sindex in range(len(embeddings[findex:])):
                thresholds.append(np.linalg.norm(embeddings[findex]-embeddings[sindex]))
                        
        return np.array(embeddings).mean(axis=0), np.mean(thresholds), anchors, patches