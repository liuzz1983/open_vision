from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import argparse
import random

import cv2
import numpy as np
from scipy import misc
import tensorflow as tf
from openvision.utils import cv_util
from openvision.utils import image_utils


from openvision.facenet.align import  align_model, detect_face

def build_model(model_dir,gpu_memory_fraction=0.3):
     with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            model = align_model.create_mtcnn(sess, model_dir)
            return model 

#args.model_dir
#img = image_utils.read_rgb(args.input_image)
# Add a random key to the filename to allow alignment using multiple processes
#random_key = np.random.randint(0, high=99999)

def align_image(model, img, margin=44, image_size=182, minsize =20,
            threshold = [ 0.6, 0.7, 0.8 ], factor = 0.709 ):
    
     # three steps's threshold
     # scale factor
    bounding_boxes, _ = detect_face.detect_face(img, minsize, model, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]
    if nrof_faces>0:
        det = bounding_boxes[:,0:4]
        img_size = np.asarray(img.shape)[0:2]

        dets = det
        rectangles = []
        for i in range(dets.shape[0]):

            #print(dets[i, :])
            det = np.squeeze(dets[i, :])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+margin/2, img_size[0])

            rectangles.append(bb)
        return rectangles
    else:
        return []

def extrace_face(img, rectangles, image_size=20):
    faces = np.zeros((len(rectangles), image_size, image_size, 3))
    for i, bb in enumerate(rectangles):
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        faces[i, :, :,:] = scaled
        
    return faces


def draw_rectangles_on_image(input_file, output_file, rectangles):
     cv_util.draw_rectangles(input_file, output_file, rectangles, thickness=1)

