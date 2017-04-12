"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
from facenet.utils import cv_util
from facenet.utils import image_utils


from facenet.align import  align_model, detect_face

def align_image(args, gpu_memory_fraction=0.3,  margin=44, image_size=182):
    
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            model = align_model.create_mtcnn(sess, args.model_dir)


    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.8 ]  # three steps's threshold
    factor = 0.709 # scale factor

    # Add a random key to the filename to allow alignment using multiple processes
    random_key = np.random.randint(0, high=99999)

    img = image_utils.read_rgb(args.input_image)

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
            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
            scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')

            rectangles.append(bb)

        cv_util.draw_rectangles(args.input_image, args.output_image, rectangles, thickness=1)

    else:
        print('Unable to align "%s"' % image_path)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_dir', type=str, default="data/align",
        help='align model path')
    parser.add_argument('input_image', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.')
    parser.add_argument('output_image', type=str,
        help='File containing the model parameters in checkpoint format.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    align_image( parse_arguments(sys.argv[1:]))
