## create script that download datasets and transform into tf-record
## Assume the datasets is downloaded into following folders
## SythTexts datasets(41G)
## data/sythtext/*

import numpy as np 
import scipy.io as sio
import os, os.path
import sys


import tensorflow as tf
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

from datasets.dataset_utils import int64_feature, float_feature, bytes_feature ,ImageCoder, norm
from datasets import icdar

from PIL import Image

data_path = sys.argv[1]


## SythText datasets is too big to store in a record. 
## So Transform tfrecord according to dir name


def _convert_to_example(image_data, shape, bbox, label,imname):
	nbbox = np.array(bbox)
	ymin = list(nbbox[:, 0])
	xmin = list(nbbox[:, 1])
	ymax = list(nbbox[:, 2])
	xmax = list(nbbox[:, 3])

	print 'shape: {}, height:{}, width:{}'.format(shape,shape[0],shape[1])
	example = tf.train.Example(features=tf.train.Features(feature={
			'image/height': int64_feature(shape[0]),
			'image/width': int64_feature(shape[1]),
			'image/channels': int64_feature(shape[2]),
			'image/shape': int64_feature(shape),
			'image/object/bbox/ymin': float_feature(ymin),
			'image/object/bbox/xmin': float_feature(xmin),
			'image/object/bbox/ymax': float_feature(ymax),
			'image/object/bbox/xmax': float_feature(xmax),
			'image/object/bbox/label': int64_feature(label),
			'image/format': bytes_feature('jpeg'),
			'image/encoded': bytes_feature(image_data),
			'image/name': bytes_feature(imname)
			}))
	return example
	

def _processing_image(label_boxes, imname,coder):
	image_data = tf.gfile.GFile(imname, 'r').read()
	image = coder.decode_jpeg(image_data)
	#image_data = np.array(Image.open(imname))
	shape = image.shape

	bbox = [l[1:] for l in label_boxes]

	label = [int(l[0]) for l in label_boxes]
	shape = list(shape)
	return image_data, shape, bbox, label, imname


def run():

	dataset = icdar.ICDAR("ch2", data_path)

	coder = ImageCoder()
	for i, img_index in enumerate(dataset.image_set_index):

		img_record = img_index + '.tfrecord'
		img_name = dataset.image_path_from_index(i)

		tf_filename = os.path.join(data_path, "tf_record", img_record)
		tfrecord_writer = tf.python_io.TFRecordWriter(tf_filename)

		labels = dataset.label_from_index(i)

		
		"""for label_index in range(labels.shape[0]):
			label = labels[label_index]
			if label[0] == -1:
				break
			image_data, shape, bbox, label ,imname= _processing_image(label, img_name, coder)
		"""
		image_data, shape, bbox, label ,imname= _processing_image(labels, img_name, coder)
		example = _convert_to_example(image_data, shape, bbox, label, imname)
		tfrecord_writer.write(example.SerializeToString())  
	print 'Transform to tfrecord finished'

if __name__ == '__main__':
	run()





