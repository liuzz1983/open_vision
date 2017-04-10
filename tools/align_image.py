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


from facenet.align import  align_model, detect_face

def read_image(image_path):
    return misc.imread(image_path)

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def draw_text(frame, text, x, y, color=(0,255,0), thickness=1, size=1):
    if x is not None and y is not None:
        cv2.putText(
            frame, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)


def align_image(args, gpu_memory_fraction=0.3,  margin=44, image_size=182):
    
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align_model.create_mtcnn(sess, args.model_dir)


    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.8 ]  # three steps's threshold
    factor = 0.709 # scale factor

    # Add a random key to the filename to allow alignment using multiple processes
    random_key = np.random.randint(0, high=99999)

    try:
        img = misc.imread(args.input_image)
    except (IOError, ValueError, IndexError) as e:
        errorMessage = '{}: {}'.format(image_path, e)
        print(errorMessage)
    else:
        if img.ndim<2:
            print('Unable to align "%s"' % image_path)
            text_file.write('%s\n' % (output_filename))
            return 
        if img.ndim == 2:
            img = facenet.to_rgb(img)
        img = img[:,:,0:3]

        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

        nrof_faces = bounding_boxes.shape[0]

        if nrof_faces>0:
            det = bounding_boxes[:,0:4]
            img_size = np.asarray(img.shape)[0:2]

            dets = det
            img2 = cv2.imread(args.input_image )
        
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

                #misc.imsave(os.path.join(output_path,str(i)+".jpeg"), scaled)
                #text_file.write('%s %d %d %d %d\n' % (output_filename, bb[0], bb[1], bb[2], bb[3]))

                bl = (bb[2], bb[3])
                tr = (bb[0], bb[1])
                color_cv = (255,0,0)
                #print(bl, tr)
                cv2.rectangle(img2, bl, tr, color=color_cv, thickness=2)
                draw_text(img2, str(bb), bb[0], bb[1])

            cv2.imwrite(args.output_image, img2)
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
