
import os
import numpy
import cv2

from config import config

class Dataset(object):
    def __init__(self):
        pass


class ImageDataSet(object):
    
    def __init__(self, data_dir, model_name, image_size):
        
        self.image_size = image_size
        self.model_name = model_name
        
        self.p_idx = 0 # positive
        self.n_idx = 0 # negative
        self.d_idx = 0 # dont care

        self.data_dir=data_dir
        self.data_dir = os.path.join(data_dir,model_name)
        self.init_dir()

        self.f1 = open(os.path.join(self.data_dir, 'pos_12.txt'), 'w')
        self.f2 = open(os.path.join(self.data_dir, 'neg_12.txt'), 'w')
        self.f3 = open(os.path.join(self.data_dir, 'part_12.txt'), 'w')

        self.img_list = open(os.path.join(self.data_dir, 'img_list.txt'), 'w')

    def init_dir(self):
        self.neg_save_dir =  os.path.join(self.data_dir,"12/negative")
        self.pos_save_dir =  os.path.join(self.data_dir,"12/positive")
        self.part_save_dir = os.path.join(self.data_dir,"12/part")
   
        for dir_path in [self.neg_save_dir,self.pos_save_dir,self.part_save_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    def save_negative_img(self, resized_im):
        self.n_idx
        save_file = os.path.join(self.neg_save_dir, "%s.jpg"%self.n_idx)
        self.f2.write("12/negative/%s"%self.n_idx + ' 0\n')
        self.img_list.write("12/negative/%s"%self.n_idx + ' 0\n')
        cv2.imwrite(save_file, resized_im)
        self.n_idx += 1

    def save_positive_img(self, resized_im,offset_x1, offset_y1, offset_x2, offset_y2):
        self.p_idx
        save_file = os.path.join(self.pos_save_dir, "%s.jpg"%self.p_idx)
        self.f1.write("12/positive/%s"%self.p_idx + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
        self.img_list.write("12/positive/%s"%self.p_idx + ' 1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
        cv2.imwrite(save_file, resized_im)
        self.p_idx += 1

    def save_part_img(self, resized_im, offset_x1, offset_y1, offset_x2, offset_y2):
        self.d_idx
        save_file = os.path.join(self.part_save_dir, "%s.jpg"%self.d_idx)
        self.f3.write("12/part/%s"%self.d_idx + ' -1 %.2f %.2f %.2f %.2f\n'%(offset_x1, offset_y1, offset_x2, offset_y2))
        cv2.imwrite(save_file, resized_im)
        self.d_idx += 1

    def close(self):
        
        self.f1.close()
        self.f2.close()
        self.f3.close()

    
class ImageLoader:
    def __init__(self, imdb, im_size, batch_size=config.BATCH_SIZE, shuffle=False):

        self.imdb = imdb
        self.batch_size = batch_size
        self.im_size = im_size
        self.shuffle = shuffle

        self.cur = 0
        self.size = len(imdb)
        self.index = np.arange(self.size)

        self.batch = None
        self.data = None
        self.label = None

        self.label_names= ['label', 'bbox_target']
        self.reset()
        self.get_batch()

    def reset(self):
        self.cur = 0
        if self.shuffle:
            np.random.shuffle(self.index)

    def iter_next(self):
        return self.cur + self.batch_size <= self.size

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.iter_next():
            self.get_batch()
            self.cur += self.batch_size
            return self.data,self.label
        else:
            raise StopIteration

    def getindex(self):
        return self.cur / self.batch_size

    def getpad(self):
        if self.cur + self.batch_size > self.size:
            return self.cur + self.batch_size - self.size
        else:
            return 0

    def get_batch(self):
        cur_from = self.cur
        cur_to = min(cur_from + self.batch_size, self.size)
        imdb = [self.imdb[self.index[i]] for i in range(cur_from, cur_to)]
        data, label = get_minibatch(imdb, self.im_size)
        self.data = data['data']
        self.label = [label[name] for name in self.label_names]


def build_dataset(annotation_file, img_base_dir, batch_size):

    with open(annotation_file, 'r') as f:
        annotations = f.readlines()

    num_images = len(annotations)
    imdb = []
    for i in range(num_images):
        annotation = annotations[i].strip().split(' ')
        image_name = annotation[0]
        im_path = os.path.join(img_base_dir, image_name)
        imdb_ = dict()
        imdb_['image'] = im_path

        label = annotation[1]
        imdb_['label'] = int(label)
        imdb_['flipped'] = False
        imdb_['bbox_target'] = np.zeros((4,))
        if len(annotation[2:]) == 4:
            bbox_target = annotation[2:]
            imdb_['bbox_target'] = np.array(bbox_target).astype(float)

        imdb.append(imdb_)   

    return imdb