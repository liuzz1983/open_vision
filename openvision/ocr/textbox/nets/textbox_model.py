
""" 
This framework is based on SSD_tensorlow(https://github.com/balancap/SSD-Tensorflow)
Add descriptions
"""

import math
from collections import namedtuple

import numpy as np
import tensorflow as tf

import tf_extended as tfe
from nets import custom_layers
from nets import textbox_common

slim = tf.contrib.slim

# =========================================================================== #
# Text class definition.
# =========================================================================== #
TextboxParams = namedtuple('TextboxParameters', 
										['img_shape',
										 'num_classes',
										 'feat_layers',
										 'feat_shapes',
										 'scale_range',
										 'anchor_ratios',
										 'normalizations',
										 'prior_scaling',
										 'step',
										 'scales'
										 ])

class TextboxNet(object):
	"""
	Implementation of the Textbox 300 network.

	The default features layers with 300x300 image input are:
	  conv4_3 ==> 38 x 38
	  fc7 ==> 19 x 19
	  conv6_2 ==> 10 x 10
	  conv7_2 ==> 5 x 5
	  conv8_2 ==> 3 x 3
	  pool6 ==> 1 x 1
	The default image size used to train this network is 300x300.
	"""
	default_params = TextboxParams(
		img_shape=(300, 300),
		num_classes=2,
		feat_layers=['conv4', 'conv7', 'conv8', 'conv9', 'conv10', 'global'],
		feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
		scale_range=[0.20, 0.90],
		anchor_ratios=[1,2,3,5,7,10],
		normalizations=[20, -1, -1, -1, -1, -1],
		prior_scaling=[0.1, 0.1, 0.2, 0.2],
		step = 0.14 ,
		scales = [0.2, 0.34, 0.48, 0.62, 0.76, 0.90]
		)

	def __init__(self, params=None):
		"""
		Init the Textbox net with some parameters. Use the default ones
		if none provided.
		"""
		if isinstance(params, TextboxParams):
			self.params = params
		else:
			self.params = self.default_params

	# ======================================================================= #
	def net(self, inputs,
			is_training=True,
			dropout_keep_prob=0.5,
			reuse=None,
			scope='text_box_300'):
		"""
		Text network definition.
		"""
		r = text_net(inputs,
					feat_layers=self.params.feat_layers,
					normalizations=self.params.normalizations,
					is_training=is_training,
					dropout_keep_prob=dropout_keep_prob,
					reuse=reuse,
					scope=scope)

		return r

	def arg_scope(self, weight_decay=0.0005, data_format='NHWC'):
		"""Network arg_scope.
		"""
		return ssd_arg_scope(weight_decay, data_format=data_format)


	def anchors(self, img_shape, dtype=np.float32):
		"""Compute the default anchor boxes, given an image shape.
		"""
		return textbox_common.textbox_achor_all_layers(img_shape,
									  self.params.feat_shapes,
									  self.params.anchor_ratios,
									  self.params.scales,
									  0.5,
									  dtype)

	def bboxes_encode(self, bboxes, anchors, num,match_threshold = 0.5,
					  scope='text_bboxes_encode'):
		"""Encode labels and bounding boxes.
		"""
		return textbox_common.tf_text_bboxes_encode(
						bboxes, anchors, num,
						matching_threshold=0.5,
						prior_scaling=self.params.prior_scaling,
						scope=scope)

	def losses(self, logits, localisations,
			   glocalisations, gscores,
			   match_threshold=0.5,
			   negative_ratio=3.,
			   alpha=1.,
			   label_smoothing=0.,
			   scope='text_box_loss'):
		"""Define the SSD network losses.
		"""
		return text_losses(logits, localisations,
						  glocalisations, gscores,
						  match_threshold=match_threshold,
						  negative_ratio=negative_ratio,
						  alpha=alpha,
						  label_smoothing=label_smoothing,
						  scope=scope)



def text_net(inputs,
			feat_layers=TextboxNet.default_params.feat_layers,
			normalizations=TextboxNet.default_params.normalizations,
			is_training=True,
			dropout_keep_prob=0.5,
			reuse=None,
			scope='text_box_300'):
	end_points = {}
	with tf.variable_scope(scope, 'text_box_300', [inputs], reuse=reuse):
		# Original VGG-16 blocks.
		net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
		end_points['conv1'] = net
		net = slim.max_pool2d(net, [2, 2], scope='pool1')
		# Block 2.
		net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
		end_points['conv2'] = net # 150,150 128
		net = slim.max_pool2d(net, [2, 2], scope='pool2')
		# Block 3. # 75 75 256
		net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
		end_points['conv3'] = net
		net = slim.max_pool2d(net, [2, 2], scope='pool3',padding='SAME')
		# Block 4. # 38 38 512
		net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
		end_points['conv4'] = net
		net = slim.max_pool2d(net, [2, 2], scope='pool4')
		# Block 5. # 19 19 512
		net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
		end_points['conv5'] = net
		net = slim.max_pool2d(net, [3, 3], stride=1, scope='pool5',padding='SAME')

		# Additional SSD blocks.
		# Block 6: let's dilate the hell out of it!
		net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
		end_points['conv6'] = net
		# Block 7: 1x1 conv. Because the fuck.
		net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
		end_points['conv7'] = net

		# Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2 (except lasts).
		end_point = 'conv8'
		with tf.variable_scope(end_point):
			net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
			net = custom_layers.pad2d(net, pad=(1, 1))
			net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
		end_points[end_point] = net
		end_point = 'conv9'
		with tf.variable_scope(end_point):
			net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
			net = custom_layers.pad2d(net, pad=(1, 1))
			net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
		end_points[end_point] = net
		end_point = 'conv10'
		with tf.variable_scope(end_point):
			net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
			net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
		end_points[end_point] = net
		end_point = 'global'
		with tf.variable_scope(end_point):
			net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
			net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
		end_points[end_point] = net

		# Prediction and localisations layers.
		predictions = []
		logits = []
		localisations = []
		for i, layer in enumerate(feat_layers):
			with tf.variable_scope(layer + '_box'):
				p, l = text_multibox_layer(layer,
										  end_points[layer],
										  normalizations[i])
			#predictions.append(prediction_fn(p))
			logits.append(p)
			localisations.append(l)

		return localisations, logits, end_points


def text_multibox_layer(layer,
					   inputs,
					   normalization=-1):
	"""
	Construct a multibox layer, return a class and localization predictions.
	The  most different between textbox and ssd is the prediction shape
	where textbox has prediction score shape (38,38,2,6)
	and location has shape (38,38,2,6,4)
	besise,the kernel for fisrt 5 layers is 1*5 and padding is (0,2)
	kernel for the last layer is 1*1 and padding is 0
	"""
	net = inputs
	if normalization > 0:
		net = custom_layers.l2_normalization(net, scaling=True)
	# Number of anchors.
	num_anchors = 6
	num_classes = 2
	# Location.
	num_loc_pred = 2*num_anchors * 4
	if(layer == 'global'):
		loc_pred = slim.conv2d(net, num_loc_pred, [1, 1], activation_fn=None, padding = 'VALID',
						   scope='conv_loc')
	else:
		loc_pred = slim.conv2d(net, num_loc_pred, [1, 5], activation_fn=None, padding = 'SAME',
						   scope='conv_loc')
	#loc_pred = custom_layers.channel_to_last(loc_pred)
	loc_pred = tf.reshape(loc_pred, loc_pred.get_shape().as_list()[:-1] + [2,num_anchors,4])
	# Class prediction.
	scores_pred = 2 * num_anchors * num_classes
	if(layer == 'global'):
		sco_pred = slim.conv2d(net, scores_pred, [1, 1], activation_fn=tf.nn.relu, padding = 'VALID',
						   scope='conv_cls')
	else:
		sco_pred = slim.conv2d(net, scores_pred, [1, 5], activation_fn=tf.nn.relu, padding = 'SAME',
						   scope='conv_cls')
	#cls_pred = custom_layers.channel_to_last(cls_pred)
	sco_pred = tf.reshape(sco_pred, sco_pred.get_shape().as_list()[:-1] + [2,num_anchors,num_classes])
	return sco_pred, loc_pred






def ssd_arg_scope(weight_decay=0.0005, data_format='NHWC'):
	"""Defines the VGG arg scope.

	Args:
	  weight_decay: The l2 regularization coefficient.

	Returns:
	  An arg_scope.
	"""
	with slim.arg_scope([slim.conv2d, slim.fully_connected],
						activation_fn=tf.nn.relu,
						weights_regularizer=slim.l2_regularizer(weight_decay),
						weights_initializer=tf.contrib.layers.xavier_initializer(),
						biases_initializer=tf.zeros_initializer()):
		with slim.arg_scope([slim.conv2d, slim.max_pool2d],
							padding='SAME',
							data_format=data_format):
			with slim.arg_scope([custom_layers.pad2d,
								 custom_layers.l2_normalization,
								 custom_layers.channel_to_last],
								data_format=data_format) as sc:
				return sc



# =========================================================================== #
# Text loss function.
# =========================================================================== #
def text_losses(logits, localisations,
			   glocalisations, gscores,
			   match_threshold=0.5,
			   negative_ratio=3.,
			   alpha=1.,
			   label_smoothing=0.,
			   scope=None):
	"""Loss functions for training the text box network.

	Arguments:
	  logits: (list of) predictions logits Tensors;
	  localisations: (list of) localisations Tensors;
	  glocalisations: (list of) groundtruth localisations Tensors;
	  gscores: (list of) groundtruth score Tensors;

	return: loss
	"""
	with tf.name_scope(scope, 'text_loss'):
		l_cross_pos = []
		l_cross_neg = []
		l_loc = []
		n_poses = 0
		for i in range(len(logits)):
			dtype = logits[i].dtype
			with tf.name_scope('block_%i' % i):
				
				# Determine weights Tensor.
				pmask = gscores[i] > match_threshold
				ipmask = tf.cast(pmask, tf.int32)
				n_pos = tf.reduce_sum(ipmask)
				fpmask = tf.cast(pmask, dtype)
				nmask = gscores[i] < match_threshold
				inmask = tf.cast(nmask, tf.int32)
				fnmask = tf.cast(nmask, dtype)
				num = tf.ones_like(gscores[i])
				n = tf.reduce_sum(num) + 1e-5
				n_poses += n_pos

				
				# Add cross-entropy loss.
				with tf.name_scope('cross_entropy_pos'):
					loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],labels=ipmask)
					#loss = tf.square(fpmask * (logits[i][:,:,:,:,:,1] - fpmask))
					loss = alpha*tf.reduce_sum(loss) / n
					l_cross_pos.append(loss)

				with tf.name_scope('cross_entropy_neg'):
					loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i],labels=inmask)
					#loss = tf.square(fnmask * (logits[i][:,:,:,:,:,0] - fnmask))
					loss = alpha*tf.reduce_sum(loss) / n
					l_cross_neg.append(loss)

				# Add localization loss: smooth L1, L2, ...
				with tf.name_scope('localization'):
					# Weights Tensor: positive mask + random negative.
					#weights = tf.expand_dims(alpha * fpmask, axis=-1)
					loss = custom_layers.abs_smooth(localisations[i] - glocalisations[i])
					loss = tf.reduce_sum(loss) / n
					l_loc.append(loss)

		# Additional total losses...
		with tf.name_scope('total'):
			total_cross_pos = tf.add_n(l_cross_pos, 'cross_entropy_pos')
			total_cross_neg = tf.add_n(l_cross_neg, 'cross_entropy_neg')
			total_cross = tf.add(total_cross_pos, total_cross_neg, 'cross_entropy')
			total_loc = tf.add_n(l_loc, 'localization')

			# Add to EXTRA LOSSES TF.collection
			tf.add_to_collection('EXTRA_LOSSES', n_poses)
			tf.add_to_collection('EXTRA_LOSSES', total_cross_pos)
			tf.add_to_collection('EXTRA_LOSSES', total_cross_neg)
			tf.add_to_collection('EXTRA_LOSSES', total_cross)
			tf.add_to_collection('EXTRA_LOSSES', total_loc)

			total_loss = tf.add(total_loc, total_cross, 'total_loss')
			tf.add_to_collection('EXTRA_LOSSES', total_loss)

		return total_loss










