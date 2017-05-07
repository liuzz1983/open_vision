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

def run_recog(align_model, facenet_model, input_image, output_dir):

	prefix_index = input_image.rfind(".")
	prefix = input_image[ : prefix_index]

	faces,rectangles = extrace_image(align_model, input_image)
	embeding = facenet_model.run(faces)

	image_dir = os.path.join(output_dir,prefix)
	mkpath(image_dir)

	embedding_file = os.path.join(output_dir, prefix+".npy")
	np.save(embedding_file,embeding)

	for i, image in enumerate(faces):
		image_file = os.path.join(image_dir, str(i)+".jpg")
		misc.imsave(image_file, image)

	image_file = os.path.join(output_dir, prefix+".jpg")

	if rectangles:
		facenet_align.draw_rectangles_on_image(input_image, image_file, rectangles)

	return embeding, faces


def main():

    args = parse_arguments(sys.argv[1:])

    align_model = facenet_align.build_model(args.align_model)
    facenet_model = load_facenet(args.facennet_model)

    for input_name in args.input_file:
    	embedding_a, faces_a = run_recog(align_model,facenet_model, input_name, args.output_dir)


if __name__ == '__main__':
    main()