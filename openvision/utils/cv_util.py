import os
import cv2


def imresample(img, sz):
    im_data = cv2.resize(img, (sz[1], sz[0]), interpolation=cv2.INTER_AREA)
    return im_data

def read_image(image_path):
    return cv2.imread(image_path)

def write_image(file_name,img):
    cv2.imwrite(file_name, img)


def draw_text(frame, text, x, y, color=(0,255,0), thickness=1, size=1):
    if x is None or y is None:
        return 
    cv2.putText( frame, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)

def draw_rectangle(img, bl, tr, color=(255, 0, 0), thickness=1):
    cv2.rectangle(img, bl, tr, color=color, thickness=2)


def resize(image, size, iterp =cv2.INTER_AREA) :
    return cv2.resize(image, size, interpolation= iterp)
    #scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')

def draw_rectangles(input_file, output_file, rectangles, color=(0,255,0), thickness=1):
    img = read_image(input_file)
    for r in rectangles:
        draw_rectangle(img, (r[0],r[1]), (r[2],r[3]), color,thickness )
    write_image(output_file, img)