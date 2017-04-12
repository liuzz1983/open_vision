## create script that download datasets and transform into tf-record
## Assume the datasets is downloaded into following folders
## SythTexts datasets(41G)
## data/sythtext/*

import numpy as np 
import scipy.io as sio
import os, os.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import tensorflow as tf
import re
from datasets.dataset_utils import int64_feature, float_feature, bytes_feature ,ImageCoder, norm

from PIL import Image

data_path = 'data/sythtext/'
os.chdir(data_path)
cellname = 'gt'
textname = 'txt'
imcell = 'imnames'
wordname = 'wordBB'
charname = 'charBB'
NUMoffolder = 200

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
			'image/name': bytes_feature(imname.tostring()),
			}))
	return example
	

def _processing_image(wordbb, imname,coder):
	image_data = tf.gfile.GFile(imname, 'r').read()
	image = coder.decode_jpeg(image_data)
	#image_data = np.array(Image.open(imname))
	shape = image.shape
	if(len(wordbb.shape) < 3 ):
		numofbox = 1
	else:
		numofbox = wordbb.shape[2]
	bbox = []
	[xmin, ymin]= np.min(wordbb,1)
	[xmax, ymax] = np.max(wordbb,1)
	xmin = np.maximum(xmin/shape[1], 0)
	ymin = np.maximum(ymin/shape[0], 0)
	xmax = np.minimum(xmax/shape[1], 1)
	ymax = np.minimum(ymax/shape[0], 1)
	if numofbox > 1:
		bbox = [[ymin[i],xmin[i],ymax[i],xmax[i]] for i in range(numofbox)] 
	if numofbox == 1:
		bbox = [[ymin,xmin,ymax,xmax]]


	label = [1 for i in range(numofbox)]
	shape = list(shape)
	return image_data, shape, bbox, label, imname


def run():
	labels = sio.loadmat('gt.mat')
	print labels.keys()
	texts = labels[textname]
	imnames = labels[imcell]
	wordBB = labels[wordname]
	charBB = labels[charname]
	coder = ImageCoder()
	for i in range(NUMoffolder):
		tf_filename = str(i+1) + '.tfrecord'
		tfrecord_writer = tf.python_io.TFRecordWriter(tf_filename)
		dir = i+1
		pattern = re.compile(r'^{}\/'.format(dir))
		i = 0
		res =[i for i in range(imnames.shape[1]) if pattern.match(imnames[0,i][0]) != None ]
		print "The size of %s folder : %s" % (dir,len(res))
		# shuffle
		res = np.random.permutation(res)
		for j in res:
			wordbb = wordBB[0,j]
			imname = imnames[0,j][0]
			image_data, shape, bbox, label ,imname= _processing_image(wordbb, imname,coder)

			example = _convert_to_example(image_data, shape, bbox, label, imname)
			tfrecord_writer.write(example.SerializeToString())  
	print 'Transform to tfrecord finished'

if __name__ == '__main__':
	run()





