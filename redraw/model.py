import tensorflow as tf
from keras import optimizers
import numpy as np
from keras import backend as K
from keras.layers import Conv2D, Dense, Activation, Reshape, Flatten
from keras.engine import Layer

from ported_ops import gen_conv

def my_gen_conv(x, neurons, activation, padding, kernel_size, strides=1, rate=1):
    #return Conv2D(neurons, activation=tf.nn.elu, padding='SAME', kernel_size=3, strides=(1, 1))(x)
    return gen_conv(x, neurons, kernel_size, strides, rate, padding, activation)

class Resize(Layer):
    def __init__(self, scale, **kwargs):
        self.scale = scale
        super(Resize, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Resize, self).build(input_shape)

    def call(self, x, method="nearest_neighbor"):
        height = tf.round(tf.cast(tf.shape(x)[1], dtype=tf.float32) * self.scale)
        width = tf.round(tf.cast(tf.shape(x)[2], dtype=tf.float32) * self.scale)

        if method == "bilinear":
            return tf.image.resize_bilinear(x, size=(height, width), align_corners=True)
        elif method == "nearest_neighbor":
            return tf.image.resize_nearest_neighbor(x, size=(height, width), align_corners=True)
        else:
            raise Exception("Unknown resize method")

    def get_output_shape_for(self, input_shape):
        height = tf_int_round(tf.cast(tf.shape(x)[1],dtype=tf.float32) * self.scale)
        width = tf_int_round(tf.cast(tf.shape(x)[2],dtype=tf.float32) * self.scale)
        return (self.input_shape[0], height, width, input_shape[3])

class WGANDiscriminator:
    def __init__(self, last_neuron_multiplier):
        base_neurons = 64
        self.l1 = Conv2D(1 * base_neurons, activation=tf.nn.leaky_relu, padding='SAME', kernel_size=5, strides=(2, 2), input_shape=(128, 128, 3))
        self.l2 = Conv2D(2 * base_neurons, activation=tf.nn.leaky_relu, padding='SAME', kernel_size=5, strides=(2, 2))
        self.l3 = Conv2D(4 * base_neurons, activation=tf.nn.leaky_relu, padding='SAME', kernel_size=5, strides=(2, 2))
        self.l4 = Conv2D(last_neuron_multiplier * base_neurons, activation=tf.nn.leaky_relu, padding='SAME', kernel_size=5, strides=(2, 2))
        self.l5 = Flatten()
        self.l6 = Dense(1)

    def attach(self, batch_data):
        x = batch_data
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        return tf.split(x, 2)


class InpaintModel:
    def create_coarse_network(self, inp, base_neurons=32):
        with tf.variable_scope("coarse_network"):
            inp = tf.concat([inp, self.ones_inp, self.ones_inp * self.mask_ph], axis=3)
            x = Conv2D(1 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=5, strides=(1, 1), input_shape=(256, 256, 3))(inp)
            x = Conv2D(2 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=3, strides=(2, 2))(x) #downsample
            x = Conv2D(2 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=3, strides=(1, 1))(x)
            x = Conv2D(4 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=3, strides=(2, 2))(x) #downsample
            x = Conv2D(4 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=3, strides=(1, 1))(x)
            x = Conv2D(4 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=3, strides=(1, 1))(x)

            downsampled_mask = tf.image.resize_nearest_neighbor(self.mask_ph, size=(x.get_shape().as_list()[1:3]), align_corners=True)
            print("orig downsampled mask shape: ", downsampled_mask.shape)

            x = Conv2D(4 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=3, dilation_rate=(2, 2))(x)
            x = Conv2D(4 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=3, dilation_rate=(4, 4))(x)
            x = Conv2D(4 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=3, dilation_rate=(8, 8))(x)
            x = Conv2D(4 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=3, dilation_rate=(16, 16))(x)
            x = Conv2D(4 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=3, strides=(1, 1))(x)
            x = Conv2D(4 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=3, strides=(1, 1))(x)

            x = Resize(2)(x) #upsample
            x = Conv2D(2 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=3, strides=(1, 1))(x)

            x = Conv2D(2 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=3, strides=(1, 1))(x)

            x = Resize(2)(x) #upsample
            x = Conv2D(1 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=3, strides=(1, 1))(x)
            
            x = Conv2D(base_neurons // 2, activation=tf.nn.elu, padding='SAME', kernel_size=3, strides=(1, 1))(x)
            x = Conv2D(3, padding='SAME', activation=None, kernel_size=3, strides=(1, 1))(x)
    
            coarse_prediction = tf.clip_by_value(x, -1., 1.)
        return coarse_prediction, downsampled_mask

    def create_fine_network(self, inp, coarse_output, downsampled_mask, batch_size, base_neurons=32):
        with tf.variable_scope("fine_network"):
            inp = coarse_output * self.mask_ph + inp * (1. - self.mask_ph)
            inp.set_shape(inp.get_shape().as_list())
            aug_x = tf.concat([inp, self.ones_inp, self.ones_inp * self.mask_ph], axis=3)

            x = Conv2D(1 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=5, strides=(1, 1))(aug_x)
            x = Conv2D(1 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=3, strides=(2, 2))(x) #downsample
            x = Conv2D(2 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=3, strides=(1, 1))(x)
            x = Conv2D(2 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=3, strides=(2, 2))(x) #downsample
            x = Conv2D(4 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=3, strides=(1, 1))(x)
            x = Conv2D(4 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=3, strides=(1, 1))(x)
            x = Conv2D(4 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=3, dilation_rate=(2, 2))(x)
            x = Conv2D(4 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=3, dilation_rate=(4, 4))(x)
            x = Conv2D(4 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=3, dilation_rate=(8, 8))(x)
            x_hallu = Conv2D(4 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=3, dilation_rate=(16, 16))(x)
            
            #attention branch 
            x = Conv2D(1 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=5, strides=(1, 1))(aug_x)
            x = Conv2D(1 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=3, strides=(2, 2))(x) #downsample
            x = Conv2D(2 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=3, strides=(1, 1))(x)
            x = Conv2D(4 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=3, strides=(2, 2))(x) #downsample
            x = Conv2D(4 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=3, strides=(1, 1))(x)
            x = Conv2D(4 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=3, strides=(1, 1))(x)
            x = Conv2D(4 * base_neurons, activation=tf.nn.relu, padding='SAME', kernel_size=3, strides=(1, 1))(x)
            x = self.contextual_attention(batch_size, x, x, downsampled_mask, 3, 1, 2) #hard coding batch size=16 for now
            x = Conv2D(4 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=3, strides=(1, 1))(x)
            x = Conv2D(4 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=3, strides=(1, 1))(x)
            x = tf.concat([x_hallu, x], axis=3)
            x = Conv2D(4 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=3, strides=(1, 1))(x)
            x = Conv2D(4 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=3, strides=(1, 1))(x)

            x = Resize(2)(x) #upsample
            x = Conv2D(2 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=3, strides=(1, 1))(x)
            x = Conv2D(2 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=3, strides=(1, 1))(x)
            x = Resize(2)(x) #upsample
            x = Conv2D(1 * base_neurons, activation=tf.nn.elu, padding='SAME', kernel_size=3, strides=(1, 1))(x)
            x = Conv2D(base_neurons // 2, activation=tf.nn.elu, padding='SAME', kernel_size=3, strides=(1, 1))(x)
            x = Conv2D(3, padding='SAME', activation=None, kernel_size=3, strides=(1, 1))(x)

        return tf.clip_by_value(x, -1., 1.)

    def gradient_penalty(self, a, b, mask, norm = 1.):
        gradients = tf.gradients(b, a)[0]
        print("gradients: ", gradients.shape)
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients) * mask, axis=[1, 2, 3]))
        penalty =  tf.reduce_mean(tf.square(slopes - norm))
        print("pen: ", penalty.shape)
        return penalty

    def contextual_attention(self, batch_size, f, b, mask, ksize, stride, rate, fuse_k=3, softmax_scale=10., training=True, fuse=True):
        with tf.variable_scope("contextual_attention"):
            print("contextual f shape: ", f.shape)
            f_shape = tf.shape(f)
            f_size = f.get_shape().as_list()
            b_size = b.get_shape().as_list()
            mask_size = mask.get_shape().as_list()

            kernel = 2*rate
            b_patches = tf.extract_image_patches(
                b, [1, kernel, kernel, 1], [1, rate*stride, rate*stride, 1], [1, 1, 1,1], padding='SAME')
            print("b_patches shape: ", b_patches.shape)
            b_patches = tf.reshape(b_patches, [-1, 32*32, kernel, kernel, b_size[3]]) # its 32 because the stride is 1*2 = 2
            b_patches = tf.transpose(b_patches, [0, 2, 3, 4, 1]) #[b, k, k, c, h*w]

            #downsizing
            f = tf.image.resize_bilinear(f, [int(f_size[1] / rate), int(f_size[2] / rate)], align_corners=True)
            b = tf.image.resize_bilinear(b, [int(b_size[1] / rate), int(b_size[2] / rate)], align_corners=True)
            mask = tf.image.resize_bilinear(mask, [int(mask_size[1] / rate), int(mask_size[2] / rate)], align_corners=True)

            sf_shape = tf.shape(f)
            sf_size = f.get_shape().as_list()
            f_batches = tf.split(f, batch_size, axis=0)

            sb_shape = tf.shape(b)
            sb_size = b.get_shape().as_list()


            sb_patches = tf.extract_image_patches(
                b, [1, kernel, kernel, 1], [1, stride, stride, 1], [1, 1, 1,1], padding='SAME')
            print("sb patches shape: ", sb_patches.shape)
            sb_patches = tf.reshape(sb_patches, [-1, 32*32, kernel, kernel, sf_size[3]])
            print("sb patches shape: ", sb_patches.shape)
            sb_patches = tf.transpose(sb_patches, [0, 2, 3, 4, 1]) #[b, k, k, c, h*w]
            print("sb patches shape: ", sb_patches.shape)

            mask_patches = tf.extract_image_patches(
                mask, [1, kernel, kernel, 1], [1, stride, stride, 1], [1, 1, 1,1], padding='SAME')

            print('input mask shape: ', mask.shape)
            print("mask patch shape: ", mask_patches.shape)
            mask_patches = tf.reshape(mask_patches, [-1, 32*32, kernel, kernel, 1])
            mask_patches = tf.transpose(mask_patches, [0, 2, 3, 4, 1]) #[b, k, k, c, h*w]
            print("patch shape: ", mask_patches.shape)
            mask_patches = mask_patches[0]
            strict_mask_patches = tf.cast(tf.equal(tf.reduce_mean(mask_patches, axis=[0,1,2], keepdims=True), 0.), tf.float32)
            
            sb_patches_batches = tf.split(sb_patches, batch_size, axis=0)
            b_patches_batches = tf.split(b_patches, batch_size, axis=0)

            conv_iden = tf.reshape(tf.eye(fuse_k), [fuse_k, fuse_k, 1, 1])

            rets = []
            for fs, sbps, bps in zip(f_batches, sb_patches_batches, b_patches_batches):
                sbps = sbps[0] #[k, k, c, h*w]
                bps = bps[0] #[k, k, c, h*w]
                print("fs: ", fs.shape)
                print("sbps: ", sbps.shape)
                sbps_normed = sbps / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(sbps), axis=[0,1,2])), 1e-4)
                x = tf.nn.conv2d(fs, sbps_normed, strides=[1,1,1,1], padding="SAME")
                
                #fusion
                x = tf.reshape(x, [1, sf_size[1] * sf_size[2], sb_size[1] * sb_size[2], 1])
                x = tf.nn.conv2d(x, conv_iden, strides=[1,1,1,1], padding='SAME') #left right consistency
                x = tf.reshape(x, [1, sf_size[1], sf_size[2], sb_size[1], sb_size[2]])
                x = tf.transpose(x, [0, 2, 1, 4, 3])
                x = tf.reshape(x, [1, sf_size[1]*sf_size[2], sb_size[1]*sb_size[2], 1])
                x = tf.nn.conv2d(x, conv_iden, strides=[1,1,1,1], padding='SAME') #top down consistency
                x = tf.reshape(x, [1, sf_size[2], sf_size[1], sb_size[2], sb_size[1]])
                x = tf.transpose(x, [0, 2, 1, 4, 3])
                
                x = tf.reshape(x, [1, sf_size[1], sf_size[2], sb_size[1]* sb_size[2]])
                print('xshape: ', x.shape)
            
                x *= strict_mask_patches
                x = tf.nn.softmax(x * softmax_scale, 3)
                x *= strict_mask_patches
                print("fshape: ", f.shape)
                print("bps shape: ", bps.shape)
                rets.append(tf.nn.conv2d_transpose(x, bps, tf.concat([[1], f_shape[1:]], axis=0), strides=[1, rate, rate, 1]) / 4.)
            rets = tf.concat(rets, axis=0) #batching them up
            rets.set_shape(f_size)
        return rets
    
    def create_wgan_discriminator(self, batch, last_neuron_multiplier):
        base_neurons = 64
        x = Conv2D(1 * base_neurons, activation=tf.nn.leaky_relu, padding='SAME', kernel_size=5, strides=(2, 2))(batch)
        x = Conv2D(2 * base_neurons, activation=tf.nn.leaky_relu, padding='SAME', kernel_size=5, strides=(2, 2))(x)
        x = Conv2D(4 * base_neurons, activation=tf.nn.leaky_relu, padding='SAME', kernel_size=5, strides=(2, 2))(x)
        x = Conv2D(last_neuron_multiplier * base_neurons, activation=tf.nn.leaky_relu, padding='SAME', kernel_size=5, strides=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(1)(x)
        return tf.split(x, 2)

    def create_wgan_loss(self, pos, neg):
        g_loss = -tf.reduce_mean(neg)
        d_loss = tf.reduce_mean(neg - pos)
        return g_loss, d_loss

    def rand_interpolate(self, batch_size, a, b):
        orig_shape = a.get_shape().as_list()
        orig_shape[0] = -1
        print("derp", orig_shape)
        i = tf.random_uniform(shape=[batch_size, 1])
        x = tf.reshape(a, [batch_size, -1])
        y = tf.reshape(b, [batch_size, -1])
        interped = x + i * (y - x)
        return tf.reshape(interped, orig_shape)

    def build_minimal_graph(self, batch_size):
        with tf.device('/gpu:0'):
            self.input_ph = tf.placeholder(np.uint8, shape=[None, 256, 256, 3])
            self.mask_ph = tf.placeholder(np.float32, shape=[1, 256, 256, 1])
            self.norm_inp = tf.div(tf.to_float(self.input_ph), 127.5) - 1.
            self.ones_inp = tf.ones_like(self.norm_inp)
            masked_batch = self.norm_inp * (1. - self.mask_ph)


            self.p_coarse, downsampled_mask = self.create_coarse_network(masked_batch)
            self.p_fine = self.create_fine_network(masked_batch, self.p_coarse, downsampled_mask, batch_size)
       
    def build_full_graph(self, batch_size=16):
        self.build_minimal_graph(batch_size)
        with tf.device('/gpu:0'):
            self.bbox_ph = tf.placeholder(np.int32, shape=[4])
            self.spatial_discounting_mask_ph = tf.placeholder(np.float32, shape=[1, 128, 128, 1])

            batch_pos = self.norm_inp
            masked_batch = batch_pos * (1. - self.mask_ph)

            fine_result = self.p_fine * self.mask_ph + masked_batch
            
            off_h = self.bbox_ph[0]
            off_w = self.bbox_ph[1]
            #targ_h = self.bbox_ph[2]
            #targ_w = self.bbox_ph[3]
            targ_h = 128
            targ_w = 128
             
            local_target = tf.image.crop_to_bounding_box(batch_pos, off_h, off_w, targ_h, targ_w)
            local_predicted = tf.image.crop_to_bounding_box(fine_result, off_h, off_w, targ_h, targ_w)
            local_fine = tf.image.crop_to_bounding_box(self.p_fine, off_h, off_w, targ_h, targ_w)
            local_coarse = tf.image.crop_to_bounding_box(self.p_coarse, off_h, off_w, targ_h, targ_w)
            local_mask = tf.image.crop_to_bounding_box(self.mask_ph, off_h, off_w, targ_h, targ_w)
            
            l1_loss = 1.2 * tf.reduce_mean(tf.abs(local_target - local_coarse) * self.spatial_discounting_mask_ph) + tf.reduce_mean(
                tf.abs(local_target - local_fine) * self.spatial_discounting_mask_ph)

            ae_loss = (1.2 * tf.reduce_mean(tf.abs(batch_pos - self.p_coarse) * (1. - self.mask_ph)) + tf.reduce_mean(
                tf.abs(batch_pos - self.p_fine) * (1. - self.mask_ph))) / tf.reduce_mean(1. - self.mask_ph)

            global_pos_neg = tf.concat([batch_pos, fine_result], axis=0)
            local_pos_neg = tf.concat([local_target, local_fine], axis=0)

            local_discriminator = WGANDiscriminator(4)
            global_discriminator = WGANDiscriminator(8)


            global_pos, global_neg = global_discriminator.attach(global_pos_neg)
            local_pos, local_neg = local_discriminator.attach(local_pos_neg)

            global_g_loss, global_d_loss = self.create_wgan_loss(global_pos, global_neg)
            local_g_loss, local_d_loss = self.create_wgan_loss(local_pos, local_neg)
            
            interp_local = self.rand_interpolate(batch_size, local_target, local_predicted)
            interp_global = self.rand_interpolate(batch_size, batch_pos, fine_result)

            dout_local = local_discriminator.attach(interp_local)
            dout_global = global_discriminator.attach(interp_global)
            
            local_gp = self.gradient_penalty(interp_local, dout_local, local_mask)
            global_gp = self.gradient_penalty(interp_global, dout_global, self.mask_ph)

            gp_loss = 10 * (local_gp + global_gp)
            
            self.total_g_losses = 0.001 * (global_g_loss + local_g_loss) + 1.2 * l1_loss + 1.2 * ae_loss
            self.total_d_losses = global_d_loss + local_d_loss + gp_loss
            
            self.train_g_op = tf.train.AdamOptimizer(1e-4, beta1=0.5, beta2=0.9).minimize(self.total_g_losses)
            self.train_d_op = tf.train.AdamOptimizer(1e-4, beta1=0.5, beta2=0.9).minimize(self.total_d_losses)

            comparison = tf.concat([self.norm_inp, masked_batch, fine_result], axis=2)

            tf.summary.scalar("loss/l1_loss", l1_loss)
            tf.summary.scalar("loss/ae_loss", ae_loss)
            tf.summary.scalar("convergence/d_loss", self.total_d_losses)
            tf.summary.scalar("convergance/local_d_loss", local_d_loss)
            tf.summary.scalar("convergance/global_d_loss", global_d_loss)
            tf.summary.scalar("wgan_loss/gp_loss", gp_loss)
            tf.summary.scalar("wgan_loss/gp_penalty_local", local_gp)
            tf.summary.scalar("wgan_loss/gp_penalty_global", global_gp)
            tf.summary.image("prediction", comparison)
            self.merged_summary = tf.summary.merge_all()
            
            self.merged_summary = tf.summary.merge_all()
    def train_g(self, sess, img, mask, bbox, spatial_discount):
        _, summary = sess.run(
            [self.train_g_op, self.merged_summary],
            feed_dict = {
                self.input_ph: img,
                self.mask_ph: mask,
                self.bbox_ph: bbox,
                self.spatial_discounting_mask_ph: spatial_discount
            }
        )
        return summary
            
    def train_d(self, sess, img, mask, bbox, spatial_discount):
        sess.run(
            [self.train_d_op],
            feed_dict = {
                self.input_ph: img,
                self.mask_ph: mask,
                self.bbox_ph: bbox,
                self.spatial_discounting_mask_ph: spatial_discount
            }
        )
