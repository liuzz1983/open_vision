from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import sys
fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, ".."))


import argparse
from distutils.dir_util import mkpath

import numpy as np
from scipy import misc
import tensorflow as tf

import openvision

from openvision.utils import tf_util
from openvision.utils import image_utils

from openvision.facenet import facenet_align
from openvision.facenet import facenet_recognization as recognization


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--align_model', type=str, default="model/align",
        help='align model path')

    parser.add_argument('--facennet_model', type=str, default="model/facenet",
        help='align model path')  

    parser.add_argument('--output_dir', type=str, default="output")

    parser.add_argument('input_file', type=str, nargs='+',
        help='Model definition. Points to a module containing the definition of the inference graph.')

    return parser.parse_args(argv)


def load_facenet(model_dir):
    sess = tf.Session()
    tf_util.load_model(sess, model_dir) 
    runner = recognization.FacenetRecongizer(sess)
    return runner

def extrace_image(model, input_name, image_size=160):
    img = image_utils.read_rgb(input_name)
    rectangles = facenet_align.align_image(model, img, minsize=20)
    faces = facenet_align.extrace_face(img,rectangles, image_size)
    return faces,rectangles

def run_recog(facenet_model,faces):
    embeding = facenet_model.run(faces)
    return embeding, faces

def extracts(args):
    align_model = facenet_align.build_model(args.align_model)

    face_sets = []
    for input_name in args.input_file:
        faces,rectangles = extrace_image(align_model, input_name)
        face_sets.append(faces)

    for i, faces in enumerate(face_sets):
        misc.imsave(str(i)+".jpg", faces[0])

    return 

def gen_embedding(args, files):

    facenet_model = load_facenet(args.facennet_model)

    embeddings = []
    for file_name in files:
        img = image_utils.read_rgb(file_name)
        prewhitened = image_utils.prewhiten(img)
        faces = np.expand_dims(prewhitened, axis=0)
        embedding_a, faces_a = run_recog(facenet_model,faces)
        embeddings.append(embedding_a[0])

    return embeddings

def main():

    args = parse_arguments(sys.argv[1:])
    
    embeddings = gen_embedding(args, args.input_file)

    emb = np.array(embeddings)
    #dist = np.dot(embeddings[0].T,embeddings[1])

    #print("image dist:", dist)

    nrof_images = len(args.input_file)
    print('Images:')
    for i in range(nrof_images):
        print('%1d: %s' % (i, args.input_file[i]))
    print('')
    
    # Print distance matrix
    print('Distance matrix')
    print('    ', end='')
    for i in range(nrof_images):
        print('    %1d     ' % i, end='')
    print('')
    for i in range(nrof_images):
        print('%1d  ' % i, end='')
        for j in range(nrof_images):
            dist = np.sqrt(np.sum(np.square(np.subtract(emb[i,:], emb[j,:]))))
            print('  %1.4f  ' % dist, end='')
        print('')
            



if __name__ == '__main__':
    main()