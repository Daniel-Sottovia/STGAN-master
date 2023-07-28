from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from functools import partial
import json
import traceback

import time
import imageio

import imlib as im
import numpy as np
import pylib
import tensorflow as tf
import tflib as tl
from PIL import Image

import data
import models

import os


# ==============================================================================
# =                                    param                                   =
# ==============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', help='experiment_name')
parser.add_argument('--gpu', type=str, default='all', help='gpu')
parser.add_argument('--dataroot', type=str, default='/data/Datasets/CelebA/Img')
# if assigned, only given images will be tested.
parser.add_argument('--img', type=int, nargs='+', default=None, help='e.g., --img 182638 202599')
# for multiple attributes
parser.add_argument('--test_atts', nargs='+', default=None)
parser.add_argument('--test_ints', nargs='+', default=None, help='leave to None for all 1')
# for single attribute
parser.add_argument('--test_int', type=float, default=1.0, help='test_int')
# for slide modification
parser.add_argument('--test_slide', action='store_true', default=False)
parser.add_argument('--n_slide', type=int, default=10)
parser.add_argument('--test_att', type=str, default=None)
parser.add_argument('--test_int_min', type=float, default=-1.0)
parser.add_argument('--test_int_max', type=float, default=1.0)
args_ = parser.parse_args()
# Foi alterado a seguir o open:
with open('D:\\Elder\\CNN\\train_test\\128\\setting.txt') as f:
    args = json.load(f)

# model
atts = args['atts']
n_att = len(atts)
img_size = args['img_size']
shortcut_layers = args['shortcut_layers']
inject_layers = args['inject_layers']
enc_dim = args['enc_dim']
dec_dim = args['dec_dim']
dis_dim = args['dis_dim']
dis_fc_dim = args['dis_fc_dim']
enc_layers = args['enc_layers']
dec_layers = args['dec_layers']
dis_layers = args['dis_layers']

label = args['label']
use_stu = args['use_stu']
stu_dim = args['stu_dim']
stu_layers = args['stu_layers']
stu_inject_layers = args['stu_inject_layers']
stu_kernel_size = args['stu_kernel_size']
stu_norm = args['stu_norm']
stu_state = args['stu_state']
multi_inputs = args['multi_inputs']
rec_loss_weight = args['rec_loss_weight']
one_more_conv = args['one_more_conv']

dataroot = args_.dataroot
img = args_.img
print('Using selected images:', img)

gpu = args_.gpu
if gpu != 'all':
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

#### testing
# multiple attributes
test_atts = args_.test_atts
test_ints = args_.test_ints
if test_atts is not None and test_ints is None:
    test_ints = [1 for i in range(len(test_atts))]
# single attribute
test_int = args_.test_int
# slide attribute
test_slide = args_.test_slide
n_slide = args_.n_slide
test_att = args_.test_att
test_int_min = args_.test_int_min
test_int_max = args_.test_int_max

thres_int = args['thres_int']
# others
use_cropped_img = args['use_cropped_img']
experiment_name = args_.experiment_name


# ==============================================================================
# =                                   graphs                                   =
# ==============================================================================

# mudança o data.Celeba
# data
sess = tl.session()

#te_data = data.Celeba(dataroot, atts, img_size, 1, part='test', sess=sess, crop=not use_cropped_img, im_no=img)
# models
Genc = partial(models.Genc, dim=enc_dim, n_layers=enc_layers, multi_inputs=multi_inputs)
Gdec = partial(models.Gdec, dim=dec_dim, n_layers=dec_layers, shortcut_layers=shortcut_layers,
               inject_layers=inject_layers, one_more_conv=one_more_conv)
Gstu = partial(models.Gstu, dim=stu_dim, n_layers=stu_layers, inject_layers=stu_inject_layers,
               kernel_size=stu_kernel_size, norm=stu_norm, pass_state=stu_state)

# inputs
xa_sample = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3])
_b_sample = tf.placeholder(tf.float32, shape=[None, n_att])
raw_b_sample = tf.placeholder(tf.float32, shape=[None, n_att])

# sample
test_label = _b_sample - raw_b_sample if label == 'diff' else _b_sample
if use_stu:
    x_sample = Gdec(Gstu(Genc(xa_sample, is_training=False),
                         test_label, is_training=False), test_label, is_training=False)
else:
    x_sample = Gdec(Genc(xa_sample, is_training=False), test_label, is_training=False)

# ==============================================================================
# =                                    test                                    =
# ==============================================================================

# initialization
ckpt_dir = 'D:\\Elder\\CNN\\train_test\\128\\checkpoints'
tl.load_checkpoint(ckpt_dir, sess)

input_image_path = 'D:\\Elder\\CNN\\train_test\\STGAN-master\\STANG-master\\imagem_teste\\20230110_155614.png'
input_image_pil = Image.open(input_image_path)
input_image_pil = input_image_pil.resize((img_size, img_size))  # resize the image if it is not of the correct dimension
input_image = np.array(input_image_pil) / 127.5 - 1  # normalize the pixel values to [-1, 1]

if input_image.shape[2] == 4:
    # Convert RGBA image to RGB
    input_image = input_image[:, :, :3]

print(len(atts))
# test
try:
    xa_sample_ipt = input_image # Alterado
    xa_sample_ipt = xa_sample_ipt[np.newaxis, ...]  # Adiciona uma dimensão extra
    # ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']
    a_sample_ipt = np.array([0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1])
    a_sample_ipt = a_sample_ipt[np.newaxis, ...]  # Adiciona uma dimensão extra
    b_sample_ipt_list = [a_sample_ipt.copy()]

    x_sample_opt_list = [xa_sample_ipt, np.full((1, img_size, img_size // 10, 3), -1.0)]
    raw_a_sample_ipt = a_sample_ipt.copy()
    raw_a_sample_ipt = (raw_a_sample_ipt * 2 - 1) * thres_int
    for i, b_sample_ipt in enumerate(b_sample_ipt_list):
        _b_sample_ipt = (b_sample_ipt * 2 - 1) * thres_int
        x_sample_opt_list.append(sess.run(x_sample, feed_dict={xa_sample: xa_sample_ipt,  # remove a new axis for batch size
                                 _b_sample: _b_sample_ipt,
                                 raw_b_sample: raw_a_sample_ipt}))

    save_dir = 'D:\\Elder\\CNN\\train_test\\STGAN-master\\STANG-master\\imagem_salva'
    pylib.mkdir(save_dir)

    # Get the current timestamp to create unique filenames
    timestamp = int(time.time())

    # Save each generated image with a unique name based on the timestamp
    for i, sample in enumerate(x_sample_opt_list):
        imageio.imwrite('%s/test_%d_%d.png' % (save_dir, timestamp, i), sample.squeeze(0))

    print('Images done!')
except:
    traceback.print_exc()
finally:
    sess.close()
