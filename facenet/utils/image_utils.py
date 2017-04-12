from scipy import misc


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def read_rgb(file_name):

    try:
        img = misc.imread(file_name)
    except (IOError, ValueError, IndexError) as e:
        errorMessage = '{}: {}'.format(image_path, e)
        print(errorMessage)
        return None 
    else:
        if img.ndim<2:
            print('Unable to align "%s"' % image_path)
            return None
        if img.ndim == 2:
            img = to_rgb(img)
        img = img[:,:,0:3]

        return img