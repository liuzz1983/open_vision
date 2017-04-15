from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse

import tensorflow as tf
import numpy as np

from openvision.facenet import facenet_recognization as recognization
from openvision.utils import tf_util


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_dir', type=str, 
        help='Directory containing the metagraph (.meta) file and the checkpoint (ckpt) file containing model parameters',
        default="var/models/20170216-091149")
    parser.add_argument('--input_img', type=str,
        help='The file containing the pairs to use for validation.', default='data/test.png')

    return parser.parse_args(argv)

def main(args):

    sess = tf.Session()
    tf_util.load_model(sess, args.model_dir) 
    runner = recognization.FacenetRecongizer(sess)

   
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

