import numpy as np
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

from skimage.measure import block_reduce
from keras.layers import InputLayer, Convolution2D
from keras.models import Sequential


def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg


def to_fully_conv(_model):

    new_model = Sequential()

    input_layer = InputLayer(input_shape=(None, None, 3), name="input_new")

    new_model.add(input_layer)
    
    for layer in _model.layers:

        if "Flatten" in str(layer):
            flattened_ipt = True
            f_dim = layer.input_shape

        elif "Dense" in str(layer):
            input_shape = layer.input_shape
            output_dim =  layer.get_weights()[1].shape[0]
            W,b = layer.get_weights()

            if flattened_ipt:
                shape = (f_dim[1],f_dim[2],f_dim[3],output_dim)
                
                new_W = W.reshape(shape)
                new_layer = Convolution2D(output_dim,
                                          (f_dim[1],f_dim[2]),
                                          strides=(1,1),
                                          activation=layer.activation,
                                          padding='valid',
                                          weights=[new_W,b],
                                          name='end')
                flattened_ipt = False

            else:
                shape = (1,1,input_shape[1],output_dim)
                new_W = W.reshape(shape)
                new_layer = Convolution2D(output_dim,
                                          (1,1),
                                          strides=(1,1),
                                          activation=layer.activation,
                                          padding='valid',
                                          weights=[new_W,b],
                                          name='end')

            new_model.add(new_layer)
        else:
            new_model.add(layer)


    return new_model

def roi_pooling(region, pooled_width, pooled_height):
    # Divide the region into non overlapping areas
    region_height = region.shape[0]
    region_width  = region.shape[1]
    h_step = region_height // pooled_height
    w_step = region_width  // pooled_width 

    areas = [[(
                i*h_step, 
                j*w_step, 
                (i+1)*h_step if i+1 < pooled_height else region_height, 
                (j+1)*w_step if j+1 < pooled_width else region_width
               ) 
               for j in range(pooled_width)] 
              for i in range(pooled_height)]

    print(areas)
    # take the maximum of each area and stack the result
    def pool_area(x): 
        return block_reduce(region[x[0]:x[2], x[1]:x[3], :], (x[2]-x[0],x[3]-x[1], 1), np.max)

    pooled_features = np.stack([[pool_area(x) for x in row] for row in areas])
    return pooled_features.squeeze()



def quantize(width, height, quantize_value):
    return width-width%quantize_value, height-height%quantize_value


def write_xml(out_path, points=[], labels=[], colors=[], rois=[], roi_labels=[], sim_values=[], roi_colors=[]):
    """
    write the dot(point) annotations in xml format of ASAP
    write the rois as rectangle annotations if provided
    returns:
        None; writes the xml output to file
    args:
        out_path is where the output xml file will be dumped to.
        each point is a 2D x and y coordinate, with a label and a color.
        points is a list of x,y coordinates (list of lists): m by 1(list of 2 coords)
        labels is a list of point labels: m by 1
        colors is a list of point colors: m by 1
        rois is a list of rectangles: n by 1; each in the shape of: [[x_min, y_min], [x_max, y_max]]
        roi_labels is a list of rectangle labels: n by 1
        roi_colors is a list of rectangle colors: n by 1
    """
    if type(points) != list:
        points = [points]
    if type(rois) != list:
        rois = [rois]

    if not points and not rois:
        raise ValueError('either points or rois should be set.')

    # the root of the xml file.
    root = ET.Element("ASAP_Annotations")

    # writing each anno one by one.
    annos = ET.SubElement(root, "Annotations")

    if not labels:
        labels = ["ROI"]*len(points)             # random label for the points if not provided.
    if not colors:
        colors = ["#000000"]*len(points)         # random color for the points if not provided.
    if not roi_labels:
        roi_labels = ["ROI"]*len(rois)       # random label for the rois if not provided.
    if not roi_colors:
        roi_colors = ["#000000"]*len(rois)   # random color for the rois if not provided.

    # writing for the rectangular ROIs
    if rois:
        for idx0, rect in enumerate(rois):
            anno = ET.SubElement(annos, "Annotation")
            anno.set("Name", "Annotation "+str(idx0) + '. sim-value: ' + str(sim_values[idx0]))
            anno.set("Type", "Polygon")
            anno.set("PartOfGroup", roi_labels[idx0])
            anno.set("Color", roi_colors[idx0])

            coords = ET.SubElement(anno, "Coordinates")
            for ridx, r in enumerate(rect):
                coord = ET.SubElement(coords, "Coordinate")
                coord.set("Order", str(ridx))
                coord.set("X", str(r[0]))
                coord.set("Y", str(r[1]))

    # writing for the dot annots
    if points:
        for idx, point in enumerate(points):
            lbl = labels[idx]
            clr = colors[idx]

            anno = ET.SubElement(annos, "Annotation")
            anno.set("Name", "Annotation "+str(idx+len(rois)))
            anno.set("Type", "Dot")
            anno.set("PartOfGroup", lbl)
            anno.set("Color", clr)

            coords = ET.SubElement(anno, "Coordinates")
            coord = ET.SubElement(coords, "Coordinate")
            coord.set("Order", "0")
            coord.set("X", str(point[0]))
            coord.set("Y", str(point[1]))

    # writing the last groups part
    anno_groups = ET.SubElement(root, "AnnotationGroups")

    # get the group names and colors from the annotations.
    # annotation labels and roi labels
    full_labels = labels+roi_labels
    # annotatoin colors and roi colors
    full_colors = colors+roi_colors
    # make the set of the labels and the colors
    labelset = list(np.unique(np.array(full_labels)))
    colorset = [full_colors[full_labels.index(l)] for l in labelset]

    for label, color in zip(labelset, colorset):
        group = ET.SubElement(anno_groups, "Group")
        group.set("Name", label)
        group.set("PartOfGroup", "None")
        group.set("Color", color)
        attr = ET.SubElement(group, "Attributes")

    # writing to the xml file with indentation
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
    with open(out_path, "w") as f:
        f.write(xmlstr)


def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes	
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by similarity of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(boxes[:,4])
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick]