
import tensorflow as tf
import numpy as np
import math





# ======================================================h===================== #
# TensorFlow implementation of Text Boxes encoding / decoding.
# =========================================================================== #

def tf_text_bboxes_encode_layer(bboxes,
                               anchors_layer, num,
                               matching_threshold=0.5,
                               prior_scaling=[0.1, 0.1, 0.2, 0.2],
                               dtype=tf.float32):
    
    """
    Encode groundtruth labels and bounding boxes using Textbox anchors from
    one layer.

    Arguments:
      bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
      anchors_layer: Numpy array with layer anchors;
      matching_threshold: Threshold for positive match with groundtruth bboxes;
      prior_scaling: Scaling of encoded coordinates.

    Return:
      (target_localizations, target_scores): Target Tensors.
    # thisi is a binary problem, so target_score and tartget_labels are same.
    """
    # Anchors coordinates and volume.

    yref, xref, href, wref = anchors_layer
    ymin = yref - href / 2.
    xmin = xref - wref / 2.
    ymax = yref + href / 2.
    xmax = xref + wref / 2. 
    vol_anchors = (xmax - xmin) * (ymax - ymin)
    
    # Initialize tensors...
    shape = (yref.shape[0], yref.shape[1], yref.shape[2], href.size)
    # all follow the shape(feat.size, feat.size, 2, 6)
    #feat_labels = tf.zeros(shape, dtype=tf.int64)
    feat_scores = tf.zeros(shape, dtype=dtype)

    feat_ymin = tf.zeros(shape, dtype=dtype)
    feat_xmin = tf.zeros(shape, dtype=dtype)
    feat_ymax = tf.ones(shape, dtype=dtype)
    feat_xmax = tf.ones(shape, dtype=dtype)

    def jaccard_with_anchors(bbox):
        """
        Compute jaccard score between a box and the anchors.
        """
        int_ymin = tf.maximum(ymin, bbox[0])
        int_xmin = tf.maximum(xmin, bbox[1])
        int_ymax = tf.minimum(ymax, bbox[2])
        int_xmax = tf.minimum(xmax, bbox[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w
        union_vol = vol_anchors - inter_vol \
            + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        jaccard = tf.div(inter_vol, union_vol)
        return jaccard
    
    """
    # never use in Textbox
    def intersection_with_anchors(bbox):
        '''
        Compute intersection between score a box and the anchors.
        '''
        int_ymin = tf.maximum(ymin, bbox[0])
        int_xmin = tf.maximum(xmin, bbox[1])
        int_ymax = tf.minimum(ymax, bbox[2])
        int_xmax = tf.minimum(xmax, bbox[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        inter_vol = h * w
        scores = tf.div(inter_vol, vol_anchors)
        return scores
    """
    
    def condition(i, feat_scores,
                  feat_ymin, feat_xmin, feat_ymax, feat_xmax):
        """Condition: check label index.
        """
        #r = tf.less(i, tf.shape(bboxes)[0])
        r = tf.less(i, num)
        return r

    def body(i, feat_scores,feat_ymin, feat_xmin, feat_ymax, feat_xmax):
        """Body: update feature labels, scores and bboxes.
        Follow the original SSD paper for that purpose:
          - assign values when jaccard > 0.5;
          - only update if beat the score of other bboxes.
        """
        # Jaccard score.
        bbox = bboxes[i]
        jaccard = jaccard_with_anchors(bbox)
        # Mask: check threshold + scores + no annotations + num_classes.
        mask = tf.greater(jaccard, feat_scores)
        mask = tf.logical_and(mask, tf.greater(jaccard, matching_threshold))
        #mask = tf.logical_and(mask, feat_scores > -0.5)
        #mask = tf.logical_and(mask, label < num_classes)
        imask = tf.cast(mask, tf.int64)
        fmask = tf.cast(mask, dtype)
        # Update values using mask.
        #feat_labels = imask * label + (1 - imask) * feat_labels
        feat_scores = tf.where(mask, jaccard, feat_scores)

        feat_ymin = fmask * bbox[0] + (1 - fmask) * feat_ymin
        feat_xmin = fmask * bbox[1] + (1 - fmask) * feat_xmin
        feat_ymax = fmask * bbox[2] + (1 - fmask) * feat_ymax
        feat_xmax = fmask * bbox[3] + (1 - fmask) * feat_xmax

        # Check no annotation label: ignore these anchors...
        #interscts = intersection_with_anchors(bbox)
        #mask = tf.logical_and(interscts > ignore_threshold,
        #                     label == no_annotation_label)
        # Replace scores by -1.
        #feat_scores = tf.where(mask, -tf.cast(mask, dtype), feat_scores)

        return [i+1, feat_scores,
                feat_ymin, feat_xmin, feat_ymax, feat_xmax]
    # Main loop definition.

    i = 0
    [i,feat_scores,
     feat_ymin, feat_xmin,
     feat_ymax, feat_xmax] = tf.while_loop(condition, body,
                                           [i, feat_scores,
                                            feat_ymin, feat_xmin,
                                            feat_ymax, feat_xmax])
    '''
    for i, bbox in enumerate(tf.unpack(bboxes, axis=0)):
        [i,feat_scores,feat_ymin, 
        feat_xmin, feat_ymax, feat_xmax] = body(i, feat_scores,
                                                feat_ymin, feat_xmin, 
                                                feat_ymax, feat_xmax,bbox)
    '''
    # Transform to center / size.
    feat_cy = (feat_ymax + feat_ymin) / 2.
    feat_cx = (feat_xmax + feat_xmin) / 2.
    feat_h = feat_ymax - feat_ymin
    feat_w = feat_xmax - feat_xmin
    # Encode features.
    feat_cy = (feat_cy - yref) / href / prior_scaling[0]
    feat_cx = (feat_cx - xref) / wref / prior_scaling[1]
    feat_h = tf.log(feat_h / href) / prior_scaling[2]
    feat_w = tf.log(feat_w / wref) / prior_scaling[3]
    # Use SSD ordering: x / y / w / h instead of ours.
    feat_localizations = tf.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1)
    return feat_localizations, feat_scores



def tf_text_bboxes_encode(bboxes,
                         anchors, num,
                         matching_threshold=0.5,
                         prior_scaling=[0.1, 0.1, 0.2, 0.2],
                         dtype=tf.float32,
                         scope='text_bboxes_encode'):
    """Encode groundtruth labels and bounding boxes using SSD net anchors.
    Encoding boxes for all feature layers.

    Arguments:
      bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
      anchors: List of Numpy array with layer anchors;
      matching_threshold: Threshold for positive match with groundtruth bboxes;
      prior_scaling: Scaling of encoded coordinates.

    Return:
      (target_labels, target_localizations, target_scores):
        Each element is a list of target Tensors.
    """

    with tf.name_scope('text_bboxes_encode'):
        target_labels = []
        target_localizations = []
        target_scores = []
        for i, anchors_layer in enumerate(anchors):
            with tf.name_scope('bboxes_encode_block_%i' % i):
                t_loc, t_scores = \
                    tf_text_bboxes_encode_layer(bboxes, anchors_layer, num,
                                                matching_threshold,
                                               prior_scaling, dtype)
                target_localizations.append(t_loc)
                target_scores.append(t_scores)
        return target_localizations, target_scores


## produce anchor for one layer
# each feature point has 12 default textboxes(6 boxes + 6 offsets boxes)
# aspect ratios = (1,2,3,5,7,10)
# feat_size :
    # conv4_3 ==> 38 x 38
    # fc7 ==> 19 x 19
    # conv6_2 ==> 10 x 10
    # conv7_2 ==> 5 x 5
    # conv8_2 ==> 3 x 3
    # pool6 ==> 1 x 1

def textbox_anchor_one_layer(img_shape,
                             feat_size,
                             ratios,
                             scale,
                             offset = 0.5,
                             dtype=np.float32):
    # Follow the papers scheme
    # 12 ahchor boxes with out sk' = sqrt(sk * sk+1)
    y, x = np.mgrid[0:feat_size[0], 0:feat_size[1]] + 0.5
    y_offset = y + offset
    y = y.astype(dtype) / feat_size[0]
    x = x.astype(dtype) / feat_size[1]
    x_offset = x
    y_offset = y_offset.astype(dtype) / feat_size[1]
    x_out = np.stack((x, x_offset), -1)
    y_out = np.stack((y, y_offset), -1)
    y_out = np.expand_dims(y_out, axis=-1)
    x_out = np.expand_dims(x_out, axis=-1)


    # 
    num_anchors = 6
    h = np.zeros((num_anchors, ), dtype=dtype)
    w = np.zeros((num_anchors, ), dtype=dtype)
    for i ,r in enumerate(ratios):
        h[i] = scale / math.sqrt(r) 
        w[i] = scale * math.sqrt(r) 
    return y_out, x_out, h, w



## produce anchor for all layers
def textbox_achor_all_layers(img_shape,
                           layers_shape,
                           anchor_ratios,
                           scales,
                           offset=0.5,
                           dtype=np.float32):
    """
    Compute anchor boxes for all feature layers.
    """
    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = textbox_anchor_one_layer(img_shape, s,
                                                 anchor_ratios,
                                                 scales[i],
                                                 offset=offset, dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors


if __name__ == "__main__":
    scales = [0.2, 0.34, 0.48, 0.62, 0.76, 0.90]
    y_out, x_out, h, w =  textbox_anchor_one_layer((300, 300), (38,38), (1,2,3,5,7,10), scale=0.2)
    print y_out.shape, x_out.shape, h.shape, w.shape

    ymin = y_out - h / 2.
    print ymin.shape


    yref, xref, href, wref =  y_out, x_out, h, w 
    ymin = yref - href / 2.
    xmin = xref - wref / 2.
    ymax = yref + href / 2.
    xmax = xref + wref / 2. 
    vol_anchors = (xmax - xmin) * (ymax - ymin)
    print href.size