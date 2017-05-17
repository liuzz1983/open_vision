import numpy as np
import os
import numpy.random as npr
import sys
import h5py

import cv2

from dataset import ImageDataSet

from utils import IoU

idx = 0
box_idx = 0

    
def gen_images(dataset, img, boxes, image_size):
    
    global box_idx

    height, width, channel = img.shape

    neg_num = 0
    while neg_num < 50:
        size = npr.randint(image_size, min(width, height) / 2)
        nx = npr.randint(0, width - size)
        ny = npr.randint(0, height - size)
        crop_box = np.array([nx, ny, nx + size, ny + size])

        Iou = IoU(crop_box, boxes)

        cropped_im = img[ny : ny + size, nx : nx + size, :]
        resized_im = cv2.resize(cropped_im, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

        if np.max(Iou) < 0.3:
            # Iou with all gts must below 0.3
            dataset.save_negative_img(resized_im)
            neg_num += 1


    for box in boxes:
        # box (x_left, y_top, x_right, y_bottom)
        x1, y1, x2, y2 = box
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        # ignore small faces
        # in case the ground truth boxes of small faces are not accurate
        if max(w, h) < 40 or x1 < 0 or y1 < 0:
            continue

        # generate negative examples that have overlap with gt
        for i in range(5):
            size = npr.randint(image_size,  min(width, height) / 2)
            # delta_x and delta_y are offsets of (x1, y1)
            delta_x = npr.randint(max(-size, -x1), w)
            delta_y = npr.randint(max(-size, -y1), h)
            nx1 = max(0, x1 + delta_x)
            ny1 = max(0, y1 + delta_y)
            if nx1 + size > width or ny1 + size > height:
                continue
            crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
            Iou = IoU(crop_box, boxes)

            cropped_im = img[ny1 : ny1 + size, nx1 : nx1 + size, :]
            resized_im = cv2.resize(cropped_im, (image_size,image_size), interpolation=cv2.INTER_LINEAR)

            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                dataset.save_negative_img(resized_im)


        print "begin to gen right "
        # generate positive examples and part faces
        for i in range(20):
            size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

            # delta here is the offset of box center
            delta_x = npr.randint(-w * 0.2, w * 0.2)
            delta_y = npr.randint(-h * 0.2, h * 0.2)

            nx1 = max(x1 + w / 2 + delta_x - size / 2, 0)
            ny1 = max(y1 + h / 2 + delta_y - size / 2, 0)
            nx2 = nx1 + size
            ny2 = ny1 + size

            if nx2 > width or ny2 > height:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])

            offset_x1 = (x1 - nx1) / float(size)
            offset_y1 = (y1 - ny1) / float(size)
            offset_x2 = (x2 - nx2) / float(size)
            offset_y2 = (y2 - ny2) / float(size)

            cropped_im = img[ny1 : ny2, nx1 : nx2, :]
            resized_im = cv2.resize(cropped_im, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

            box_ = box.reshape(1, -1)
            print "begin --------",crop_box, box_, IoU(crop_box, box_)
            if IoU(crop_box, box_) >= 0.65:
                dataset.save_positive_img(resized_im, offset_x1, offset_y1, offset_x2, offset_y2)
            elif IoU(crop_box, box_) >= 0.4:
                dataset.save_part_img(resized_im, offset_x1, offset_y1, offset_x2, offset_y2)

        box_idx += 1
        #print "%s images done, pos: %s part: %s neg: %s"%(idx, p_idx, d_idx, n_idx)

def parse_annotations(anno_file,image_base_dir):
    
    with open(anno_file, 'r') as f:
        annotations = f.readlines()

    num = len(annotations)
    index = 0

    print "%d pics in total" % num
    while annotations :
        annotations = annotations[index:]

        image_name = annotations[0].strip()
        face_num = int(annotations[1])

        im_path = os.path.join(image_base_dir,image_name+".jpg")
        boxes = []
        for i in range(face_num):
            values = annotations[i+2].strip().split(' ')
            print values
            values = map(float, values[:5])
            left_x, right_x = int(values[3] - values[0]), int(values[3] + values[0])
            top_y, bottom_y = int(values[4] - values[1]), int(values[4] + values[1])
            boxes.append( [left_x, top_y, right_x, bottom_y] )

        
        boxes_ = np.array(boxes, dtype=np.int)

        print "------", im_path,boxes_
        img = cv2.imread(im_path)

        yield img, boxes_

        annotations = annotations[face_num+2:]


def parse_annotations_files(dataset,image_base_dir, files):
    for anno_file in files:
        for img, bboxes in parse_annotations(anno_file, image_base_dir):
            gen_images(dataset, img, bboxes, dataset.image_size)

if __name__ == "__main__":
    
    model_name = sys.argv[1]
    image_size = int(sys.argv[2])
    data_dir = sys.argv[3]

    dataset = ImageDataSet(data_dir, model_name, image_size)
    parse_annotations_files(dataset, sys.argv[4],sys.argv[5:])

