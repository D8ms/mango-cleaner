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
    img_dir = "/home/tqi/work/shared_data/datasets/celeba/training/"
    imgs = []
    for i in range(5000):
        idx = str(random.randint(0, 199999)).rjust(6, "0")

        path = img_dir + idx + ".jpg"
        img = cv2.imread(path)
        img = cv2.resize(img, (256, 256)) #turns into (256, 256, 3)
        imgs.append(img)
        print(i)
    return imgs

def get_batch(imgs, n):
    ret = []
    l = len(imgs)
    for i in range(n):
        j = random.randint(0, l - 1)
        img = imgs[j]
        ret.append(img)
    return ret

def train_g(sess, imgs, model):
    batch_imgs = get_batch(imgs, BATCH_SIZE)
    bbox = rand_bbox()
    spatial_discount = simple_discounting_mask()
    mask = bbox_to_mask(bbox)
    summary = model.train_g(sess, batch_imgs, mask, bbox, spatial_discount)
    return summary

def train_d(sess, imgs, model):
    batch_imgs = get_batch(imgs, BATCH_SIZE)
    bbox = rand_bbox()
    spatial_discount = simple_discounting_mask()
    mask = bbox_to_mask(bbox)
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
            print("epoch:", str(i))
            print(imgs[0].shape)
            sys.stdout.flush()
            for _ in range(5):
                train_d(sess, imgs, model)
            summary = train_g(sess, imgs, model)
            for _ in range(5):
                train_d(sess, imgs, model)
            summary_recorder.add_summary(summary, i)
if __name__ == "__main__":
    run()
