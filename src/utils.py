import numpy as np

def rle_decode(mask_rle,shape=(768, 768)):
    '''
    ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
    :param mask_rle: run-length as string formated (start length)
    :param shape:(height,width) of array to return
    :return: numpy array. 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x,dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1] ,dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def mask_as_image(masks):
    '''
    :param masks:
    :param shape:
    :return:
    '''
    all_masks = np.zeros((768, 768),dtype=np.float32)
    for mask in masks:
        if isinstance(mask,str):
            all_masks += rle_decode(mask)
    return all_masks 
    # return np.expand_dims(all_masks,-1)