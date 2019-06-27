import random
import sys
from crop_util import *
from model import InpaintModel
import glob
from PIL import Image
import tensorflow as tf

CROP_W_MARGIN = 50
CROP_H_MARGIN = 140
EPOCHS = 50000
BATCH_SIZE = 16

def load_images():
    img_dir = "/home/tqi/work/deep_clean/redraw/ss_data/*.png"
    full_paths = glob.glob(img_dir)
    imgs = []
    
    for img_path in full_paths:
        img = Image.open(img_path)
        width, height = img.size
        img.resize((1024, 685), Image.BILINEAR)
        arr = np.array(img)
        print(arr.shape)
        #if arr.shape[-1] == 3:
        #    r = arr[:,:,0]
        #    g = arr[:,:,1]
        #    b = arr[:,:,2]
        #    arr = 0.2989 * r + 0.5870 * g + 0.1140 * b
        imgs.append(arr)
    return imgs

def get_crops(imgs, n):
    ret = []
    l = len(imgs)
    for i in range(n):
        j = random.randint(0, l - 1)
        img = imgs[j]
        ret.append(rand_crop(img, 256, CROP_W_MARGIN, CROP_H_MARGIN))
    return ret

def train_g(sess, imgs, model):
    batch_imgs = get_crops(imgs, BATCH_SIZE)
    bbox = rand_bbox()
    mask, spatial_discount = bbox_to_many_masks(bbox, 1)
    #spatial_discount = simple_discounting_mask()
    #mask = bbox_to_mask(bbox)
    summary = model.train_g(sess, batch_imgs, mask, bbox, spatial_discount)
    return summary

def train_d(sess, imgs, model):
    batch_imgs = get_crops(imgs, BATCH_SIZE)
    bbox = rand_bbox()
    mask, spatial_discount = bbox_to_many_masks(bbox, 1)
    #spatial_discount = simple_discounting_mask()
    #mask = bbox_to_mask(bbox)
    model.train_d(sess, batch_imgs, mask, bbox, spatial_discount)

def run():
    imgs = load_images()
    tf.config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    with tf.Session(config=tf.config) as sess:
        model = InpaintModel()
        model.build_full_graph()
        summary_recorder = tf.summary.FileWriter("summary", sess.graph)
        sess.run(tf.global_variables_initializer())
        for i in range(EPOCHS):
            #yeah d trains twice per epoch
            print("epoch:", str(i))
            sys.stdout.flush()
            for _ in range(5):
                train_d(sess, imgs, model)
            summary = train_g(sess, imgs, model)
            for _ in range(5):
                train_d(sess, imgs, model)
            summary_recorder.add_summary(summary, i)
if __name__ == "__main__":
    run()
