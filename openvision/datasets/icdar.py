from __future__ import print_function
import os
import numpy as np

import cv2
import re
import codecs


class ICDAR(object):
    """
    Implementation of Imdb for Pascal VOC datasets

    Parameters:
    ----------
    image_set : str
        set to be used, can be train, val, trainval, test
    data_path : str
        devkit path of VOC dataset
    shuffle : boolean
        whether to initial shuffle the image list
    is_train : boolean
        if true, will load annotations
    """
    def __init__(self, image_set, data_path, shuffle=False, is_train=True):

        self.image_set = image_set
        self.data_path = data_path
        self.extension = '.jpg'
        self.is_train = is_train

        self.classes = ['text', ]

        self.config = {#'use_difficult': True,
                       'comp_id': 'comp4',
                       'padding': 56}

        print(len(self.classes))
        self.num_classes = len(self.classes)
        self.image_set_index = self._load_image_set_index(shuffle)
        self.num_images = len(self.image_set_index)
        if self.is_train:
            self.labels = self._load_image_labels()
    '''
    @property
    def cache_path(self):
        """
        make a directory to store all caches

        Returns:
        ---------
            cache path
        """
        cache_path = os.path.join(os.path.dirname(__file__), '..', 'cache')
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        return cache_path
    '''
    def _load_image_set_index(self, shuffle):
        """
        find out which indexes correspond to given image set (train or val)

        Parameters:
        ----------
        shuffle : boolean
            whether to shuffle the image list
        Returns:
        ----------
        entire list of images specified in the setting
        """
        image_set_index_file = os.path.join(self.data_path, self.image_set + '.txt')
        assert os.path.exists(image_set_index_file), 'Path does not exist: {}'.format(image_set_index_file)
        with open(image_set_index_file) as f:
            image_set_index = [x.strip() for x in f.readlines()]
	#print(type(image_set_index))
        if shuffle:
            np.random.shuffle(image_set_index)
        return image_set_index
    
    def image_path_from_index(self, index):
        """
        given image index, find out full path

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        full path of this image
        """
        assert self.image_set_index is not None, "Dataset not initialized"
        name = self.image_set_index[index]
        image_file = os.path.join(self.data_path, self.image_set, name + self.extension)
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file
    

    def _image_path_from_index(self, index):
        """
        given image index, find out annotation path

        Parameters:
        ----------
        index: int
            index of a specific image

        Returns:
        ----------
        full path of annotation file
        """
        image_file = os.path.join(self.data_path, self.image_set, index + self.extension)
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file

    def label_from_index(self, index):
        """
        given image index, return preprocessed ground-truth

        Parameters:
        ----------
        index: int
            index of a specific image
        Returns:
        ----------
        ground-truths of this image
        """
        assert self.labels is not None, "Labels not processed"
        return self.labels[index, :, :]
        #return self.labels[index]

    def _label_path_from_index(self, index):
        """
        given image index, find out annotation path

        Parameters:
        ----------
        index: int
            index of a specific image

        Returns:
        ----------
        full path of annotation file
        """
        label_file = os.path.join(self.data_path, self.image_set + '_GT', 'gt_' + index + '.txt')
        assert os.path.exists(label_file), 'Path does not exist: {}'.format(label_file)
        return label_file

    def _load_image_labels(self):
        """
        preprocess all ground-truths

        Returns:
        ----------
        labels packed in [num_images x max_num_objects x 5] tensor
        """
        temp = []
        max_objects = 0

        # load ground-truth from xml annotations
        for idx in self.image_set_index:
            image_file = self._image_path_from_index(idx)
            height, width, channels = cv2.imread(image_file).shape
            width = float(width)
            height = float(height)
            label = []
            label_file = self._label_path_from_index(idx)
            for line in codecs.open(label_file,"r",encoding="utf-8-sig"):
                line = line.strip().split()
                m = []
                for i in xrange(len(line) - 1):
                    #print line[i]
                    m.append(re.match(r'[0-9]+', line[i]).group())
                xmin = float(m[0]) / width
                ymin = float(m[1]) / height
                xmax = float(m[2]) / width
                ymax = float(m[3]) / height
                cls_id = self.classes.index('text')
                label.append([cls_id, xmin, ymin, xmax, ymax])

            temp.append(np.array(label))
            max_objects = max(max_objects, len(label))

        # add padding to labels so that the dimensions match in each batch
        # TODO: design a better way to handle label padding
        assert max_objects > 0, "No objects found for any of the images"
        assert max_objects <= self.config['padding'], "# obj exceed padding"

        self.padding = self.config['padding']
        labels = []
        for label in temp:
            label = np.lib.pad(label, ((0, self.padding-label.shape[0]), (0,0)), \
                               'constant', constant_values=(-1, -1))
            labels.append(label)

        #return labels
        return np.array(labels)

