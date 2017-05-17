#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import datetime
import numpy as np
import argparse
import cv2

import tensorflow as tf

from config import config
from model import *
from dataset import ImageLoader, build_dataset

def build_optimizer(base_lr,loss,data_num, batch_size = 128, lr_epoch =  [8, 14]):
    
    lr_factor=0.1
    global_step = tf.Variable(0, trainable=False)

    boundaries=[ int(epoch * data_num/batch_size) for epoch in lr_epoch]

    lr_values=[ base_lr*(lr_factor**x) for x in range(0,len(lr_epoch)+1)]
    lr_op=tf.train.piecewise_constant(global_step, boundaries, lr_values)

    optimizer=tf.train.MomentumOptimizer(lr_op,0.9)
    train_op=optimizer.minimize(loss,global_step)

    return train_op,lr_op

def compute_accuracy(cls_prob,label):
    keep=(label>=0)
    pred=np.zeros_like(cls_prob)
    pred[cls_prob>0.5]=1
    return np.sum(pred[keep]==label[keep])*1.0/np.sum(keep)


def get_minibatch(imdb, im_size):
    # im_size: 12, 24 or 48
    num_images = len(imdb)
    processed_ims = list()
    cls_label = list()
    bbox_reg_target = list()
    for i in range(num_images):
        #print imdb[i]['image']
        im = cv2.imread(imdb[i]['image']+".jpg")
        h, w, c = im.shape
        cls = imdb[i]['label']
        bbox_target = imdb[i]['bbox_target']

        assert h == w == im_size, "image size wrong"
        if imdb[i]['flipped']:
            im = im[:, ::-1, :]

        im_tensor = im/127.5
        processed_ims.append(im_tensor)
        cls_label.append(cls)
        bbox_reg_target.append(bbox_target)

    im_array = np.asarray(processed_ims)
    label_array = np.array(cls_label)
    bbox_target_array = np.vstack(bbox_reg_target)
    '''
    bbox_reg_weight = np.ones(label_array.shape)
    invalid = np.where(label_array == 0)[0]
    bbox_reg_weight[invalid] = 0
    bbox_reg_weight = np.repeat(bbox_reg_weight, 4, axis=1)
    '''

    data = {'data': im_array}
    label = {'label': label_array,
             'bbox_target': bbox_target_array}

    return data, label



def train_net(net_factory, prefix, end_epoch, dataset_path, img_base_dir, net=12, frequent=50, base_lr=0.01):

    batch_size = 128

    img_dataset = build_dataset(dataset_path,img_base_dir, batch_size)
    dataset_num  = len(img_dataset)

    train_data = ImageLoader(img_dataset,net)

    input_image=tf.placeholder(tf.float32, shape=[batch_size,net,net,3],name='input_image')
    label=tf.placeholder(tf.float32, shape=[batch_size], name='label')
    bbox_target=tf.placeholder(tf.float32, shape=[batch_size,4],name='bbox_target')

    cls_prob_op, bbox_pred_op, cls_loss_op, bbox_loss_op=net_factory(input_image, label, bbox_target)

    train_op, lr_op = build_optimizer(base_lr,cls_loss_op+bbox_loss_op, dataset_num)

    sess=tf.Session()
    saver=tf.train.Saver()

    sess.run(tf.global_variables_initializer())


    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('./data/logs', sess.graph)

    total_step  = 0
    for cur_epoch in range(1,end_epoch+1):
        
        train_data.reset()

        accuracy_list=[]
        cls_loss_list=[]
        bbox_loss_list=[]
        
        for batch_idx,(image_x,(label_y,bbox_y)) in enumerate(train_data):
            total_step += 1
            # train the dataset
            sess.run(train_op, feed_dict={
                    input_image:image_x,
                    label:label_y,
                    bbox_target:bbox_y
                    })
            
            if batch_idx%frequent == 0:
                cls_pred,cls_loss,bbox_loss,lr = sess.run(
                    [ cls_prob_op,cls_loss_op,bbox_loss_op,lr_op ],
                    feed_dict={ 
                            input_image:image_x,
                            label:label_y,
                            bbox_target:bbox_y
                            }
                    )

                accuracy= compute_accuracy(cls_pred,label_y)
                print "%s : Epoch: %d, Step: %d, accuracy: %3f, cls loss: %4f, bbox loss: %4f,lr:%f "% \
                    (datetime.datetime.now(),cur_epoch,batch_idx,accuracy,cls_loss,bbox_loss,lr)

                #summary_str = sess.run(merged_summary_op)
                #summary_writer.add_summary(summary_str, total_step)
                
                accuracy_list.append(accuracy)
                cls_loss_list.append(cls_loss)
                bbox_loss_list.append(bbox_loss)

        print "Epoch: %d, accuracy: %3f, cls loss: %4f, bbox loss: %4f "% \
            (cur_epoch,np.mean(accuracy_list),np.mean(cls_loss_list),np.mean(bbox_loss_list))

        saver.save(sess,prefix,cur_epoch)

def train_mtcnn_net(model, image_set, 
                root_path, 
                dataset_path, prefix,
                end_epoch, frequent, lr):


    if model == "pnet":
        sym = P_Net
        net = 12
    elif model == "onet":
        sym = O_Net
        net = 24
    elif model == "rnet":
        sym = R_Net
        net = 24
        
    train_net(sym, prefix, end_epoch, dataset_path, image_set, net, frequent, lr)

data_name = "data/pnet/pnet/img_list.txt "
def parse_args():
    parser = argparse.ArgumentParser(description='Train proposal net(12-net)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model_name', dest='model_name', help='training set',
                        default='pnet', type=str)
    parser.add_argument('--image_set', dest='image_set', help='training set',
                        default='data/pnet/', type=str)
    parser.add_argument('--root_path', dest='root_path', help='output data folder',
                        default='./data', type=str)
    parser.add_argument('--dataset_path', dest='dataset_path', help='dataset folder',
                        default="data/pnet/pnet/img_list.txt", type=str)

    parser.add_argument('--prefix', dest='prefix', help='new model prefix',
                        default="./data/model/pnet", type=str)
    parser.add_argument('--end_epoch', dest='end_epoch', help='end epoch of training',
                        default=16, type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
                        default=200, type=int)
    parser.add_argument('--learning_rate', dest='lr', help='learning rate',
                        default=0.01, type=float)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print 'Called with argument:'
    print args
    train_mtcnn_net(args.model_name, args.image_set, args.root_path, args.dataset_path, args.prefix,
                args.end_epoch, args.frequent, args.lr)