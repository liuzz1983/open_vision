import load_batch
import tensorflow as tf
import os
import sys
from datasets import sythtextprovider
from processing import txt_preprocessing
import tf_utils

slim = tf.contrib.slim
from nets import txtbox_300

dataset_dir = "data/ch2/tf_record"
num_readers = 2
batch_size = 3
is_training = True
num_preprocessing_threads = 2

#out_shape = net.params.img_shape
#anchors = net.anchors(out_shape)
net = txtbox_300.TextboxNet()
out_shape = net.params.img_shape
anchors = net.anchors(out_shape)

dataset = sythtextprovider.get_datasets(dataset_dir)

provider = slim.dataset_data_provider.DatasetDataProvider(
			dataset,
			num_readers=num_readers,
			common_queue_capacity=20 * batch_size,
			common_queue_min=10 * batch_size,
			shuffle=True)

[image, shape, glabels, gbboxes] = provider.get(['image', 'shape',
										 'object/label',
										 'object/bbox'])


image, glabels, gbboxes,num = txt_preprocessing.preprocess_image(image,  glabels,gbboxes, 
											out_shape,is_training=is_training)

glocalisations, gscores = net.bboxes_encode( gbboxes, anchors, num,match_threshold = 0.5)

batch_shape = [1] + [len(anchors)] * 2


r = tf.train.batch(
	tf_utils.reshape_list([image, glocalisations, gscores]),
	batch_size=batch_size,
	num_threads=num_preprocessing_threads,
	capacity=5 * batch_size)

b_image, b_glocalisations, b_gscores= tf_utils.reshape_list(r, batch_shape)


print image
print b_image


tf.Graph().as_default()
arg_scope = net.arg_scope(weight_decay=0.00004)
slim.arg_scope(arg_scope)
img = tf.placeholder(tf.float32, [ -1,300, 300, 3])
print img 
localisations, logits, end_points = net.net(img, is_training=True)
print localisations, logits


