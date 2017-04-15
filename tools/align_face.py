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
import numpy as np
from scipy import misc

from openvision.utils import image_utils
from openvision.facenet import facenet_align


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_dir', type=str, default="data/align",
        help='align model path')
    parser.add_argument('input_image', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.')
    parser.add_argument('output_image', type=str,
        help='File containing the model parameters in checkpoint format.')
    return parser.parse_args(argv)


def main():
    args = parse_arguments(sys.argv[1:])

    model = facenet_align.build_model(args.model_dir)
    img = image_utils.read_rgb(args.input_image)

    rectangles = facenet_align.align_image(model, img)

    if rectangles:
        facenet_align.draw_rectangles_on_image(args.input_image, args.output_image, rectangles)
    else:
        print("cant not find face in image")

    faces = facenet_align.extrace_face(img,rectangles, image_size= 40)

    #for i, face in enumerate(faces):
    #    misc.imsave(str(i)+".jpg", face)


if __name__ == '__main__':
    main()

