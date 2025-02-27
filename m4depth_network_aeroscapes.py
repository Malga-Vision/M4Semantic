"""
----------------------------------------------------------------------------------------
Copyright (c) 2022 - Michael Fonder, University of Liège (ULiège), Belgium.

This program is free software: you can redistribute it and/or modify it under the terms
of the GNU Affero General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License along with this
program. If not, see < [ https://www.gnu.org/licenses/ | https://www.gnu.org/licenses/ ] >.
----------------------------------------------------------------------------------------
"""

import tensorflow as tf
from tensorflow import keras as ks
from utils.depth_operations import *
from collections import namedtuple
from focal_loss import SparseCategoricalFocalLoss

M4depthAblationParameters = namedtuple('M4depthAblationParameters', ('DINL', 'SNCV', 'time_recurr', 'normalize_features', 'subdivide_features', 'level_memory'),
                                    defaults=(True, True, True, True, True, True))

'''
def dice_coef(y_true, y_pred, M=12, smooth=1.0):
    
    dice = 0
    for index in range(M):
        #print("INDEX = ", index)
        #print("Y_TRUE = ", y_true[:,:,:,index]) 
        #print("Y_PRED = ", y_pred[:,:,:,index])      
        y_true_f = tf.cast(tf.reshape(y_true[:,:,:,index], [-1]),'float32')
        y_pred_f = tf.cast(tf.reshape(y_pred[:,:,:,index], [-1]),'float32')
        #y_pred_f = tf.where(y_pred_f >= 0.9, 1.0, y_pred_f)
        #y_pred_f = tf.where(y_pred_f <= 0.08, 0.0, y_pred_f)
        #print("INDEX = ", index)
        intersection = tf.reduce_sum(tf.math.multiply(y_true_f,y_pred_f))
        #print("INTERSECTION = ", intersection)
        dice_val = ((2. * intersection) + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
        dice = dice + dice_val
        #print("DICE = ", dice_val)
    return dice
    
def dice_coef_loss(y_true, y_pred, M=12):
    return M - dice_coef(y_true, y_pred, M)
'''

class DomainNormalization(ks.layers.Layer):
    # Normalizes a feature map according to the procedure presented by
    # Zhang et.al. in "Domain-invariant stereo matching networks".

    def __init__(self, regularizer_weight=0.0004):
        super(DomainNormalization, self).__init__()
        self.regularizer_weight = regularizer_weight

    def build(self, input_shape):
        channels = input_shape[-1]

        self.scale = self.add_weight(name="scale", shape=[1, 1, 1, channels], dtype='float32',
                                     initializer=tf.ones_initializer(), trainable=True)
        self.bias = self.add_weight(name="bias", shape=[1, 1, 1, channels], dtype='float32',
                                    initializer=tf.zeros_initializer(), trainable=True)

        # Add regularization loss on the scale factor
        regularizer = tf.keras.regularizers.L2(self.regularizer_weight)
        self.add_loss(regularizer(self.scale))

    def call(self, f_map):
        mean = tf.math.reduce_mean(f_map, axis=[1, 2], keepdims=True, name=None)
        var = tf.math.reduce_variance(f_map, axis=[1, 2], keepdims=True, name=None)
        normed = tf.math.l2_normalize((f_map - mean) / (var + 1e-12), axis=-1)
        return self.scale * normed + self.bias


class FeaturePyramid(ks.layers.Layer):
    # Encoder of the network
    # Builds a pyramid of feature maps.

    def __init__(self, settings, regularizer_weight=0.0004, trainable=True):
        super(FeaturePyramid, self).__init__(trainable=trainable)

        self.use_dinl = settings["ablation"].DINL
        self.out_sizes = [16, 32, 64, 96, 128, 192][:settings["nbre_lvls"]]

        init = ks.initializers.HeNormal()
        reg = ks.regularizers.L1(l1=regularizer_weight)
        self.conv_layers_s1 = [ks.layers.Conv2D(
            nbre_filters, 3, strides=(1, 1), padding='same',
            kernel_initializer=init, kernel_regularizer=reg)
            for nbre_filters in self.out_sizes
        ]
        self.conv_layers_s2 = [ks.layers.Conv2D(
            nbre_filters, 3, strides=(2, 2), padding='same',
            kernel_initializer=init, kernel_regularizer=reg)
            for nbre_filters in self.out_sizes
        ]

        self.dn_layers = [DomainNormalization(regularizer_weight=regularizer_weight) for nbre_filters in self.out_sizes]

    @tf.function  # (jit_compile=True)
    def call(self, images):
        feature_maps = images
        outputs = []
        for i, (conv_s1, conv_s2, dn_layer) in enumerate(zip(self.conv_layers_s1, self.conv_layers_s2, self.dn_layers)):
            tmp = conv_s1(feature_maps)
            if self.use_dinl and i == 0:
                tmp = dn_layer(tmp)
            tmp = tf.nn.leaky_relu(tmp, 0.1)

            tmp = conv_s2(tmp)
            feature_maps = tf.nn.leaky_relu(tmp, 0.1)
            outputs.append(feature_maps)

        return outputs


class SemanticRefiner(ks.layers.Layer):
    # Sub-network in charge of refining an input parallax estimate
    # (name to be kept to keep backward compatibility with existing trained weights)

    def __init__(self, regularizer_weight=0.0004, num_classes = 14):
        super(SemanticRefiner, self).__init__()

        init = ks.initializers.HeNormal()
        reg = ks.regularizers.L1(l1=regularizer_weight)
        self.classes = num_classes
        
        conv_channels = [128, 128, 96]
        self.prep_conv_layers = [ks.layers.Conv2D(
            nbre_filters, 3, strides=(1, 1), padding='same',
            kernel_initializer=init, kernel_regularizer=reg)
            for nbre_filters in conv_channels
        ]
        #conv_channels = [64, 32, 16, 5]
        conv_channels = [64, 32, 16, self.classes+4]
        self.est_s_conv_layers = [ks.layers.Conv2D(
            nbre_filters, 3, strides=(1, 1), padding='same',
            kernel_initializer=init, kernel_regularizer=reg)
            for nbre_filters in conv_channels
        ]

    @tf.function
    def call(self, feature_map):

        prev_out = tf.identity(feature_map)

        for i, conv in enumerate(self.prep_conv_layers):
            prev_out = conv(prev_out)
            prev_out = tf.nn.leaky_relu(prev_out, 0.1)

        prev_outs = [prev_out, prev_out] ## Why 2?

        for i, convs in enumerate(zip(self.est_s_conv_layers)):

            for j, (prev, conv) in enumerate(zip(prev_outs, convs)):
                prev_outs[j] = conv(prev)

                if i < len(self.est_s_conv_layers) - 1:  # Don't activate last convolution output
                    prev_outs[j] = tf.nn.leaky_relu(prev_outs[j], 0.1)
                else:
                    outs1 = tf.nn.softmax(prev_outs[j][:,:,:,:self.classes])
                    outs2 = prev_outs[j][:,:,:,self.classes:]
                    prev_outs[j] = tf.concat([outs1, outs2], axis = 3)
                ### We should apply softmax activation on the last convolutional output (only till channel 13)

        return prev_outs # tf.concat(prev_outs, axis=-1)
        
class SemanticEstimatorLevel(ks.layers.Layer):
    # Stackable level for the decoder of the architecture
    # Outputs both a depth and a parallax map

    def __init__(self, settings, depth, regularizer_weight=0.0004):
        super(SemanticEstimatorLevel, self).__init__()

        self.is_training = settings["is_training"]
        self.ablation = settings["ablation"]
        self.classes = settings["classes"] # Always assumes the last class index represents other objects or void

        self.semantic_refiner = SemanticRefiner(regularizer_weight=regularizer_weight, num_classes = self.classes)
        self.init = True
        self.lvl_depth = depth  # 1, 2, 3, 4, 5, 6 
        self.lvl_mul = depth-3   # -2, -1, 0, 1, 2, 3

    def build(self, input_shapes):
        # Init. variables required to store the state of the level between two time steps when working in an online fashion
        self.shape = input_shapes

        f_maps_init = tf.zeros_initializer()
        s_maps_init = tf.ones_initializer()
        if (not self.is_training):
            #@@@@@@@@@@@@@@@
            ## Can be removed because we don't use temporal info in semantic segmentation simplified arch
            self.prev_f_maps_time = self.add_weight(name="prev_f_maps", shape=self.shape, dtype='float32',
                                               initializer=f_maps_init, trainable=False, use_resource=False)
            ## This is  not correct for semantic
            ## Be careful where this is used later
            self.semantic_prev_time = self.add_weight(name="semantic_prev_t", shape=self.shape[:3] + [self.classes], dtype='float32',
                                                initializer=s_maps_init, trainable=False, use_resource=False)
            #@@@@@@@@@@@@@@@
        else:
            print("Skipping temporal memory instanciation")

    @tf.function
    def call(self, curr_f_maps_lvl, prev_s_est_lvl, curr_depth_lvl, rot, trans, camera, new_traj, prev_f_maps_time=None, prev_semantic_time=None):
        #print("PREV_SEMANTIC_TIME= ", prev_semantic_time)
        with tf.name_scope("SemanticEstimator_lvl"):
            b, h, w, c = self.shape

            # Disable feature vector subdivision if required
            if self.ablation.subdivide_features:
                nbre_cuts = 2**(self.lvl_depth//2) ## K
            else:
                nbre_cuts = 1

            # Disable feature vector normalization if required
            if self.ablation.normalize_features:
                vector_processing = lambda f_map : tf.linalg.normalize(f_map, axis=-1)[0]
            else:
                vector_processing = lambda f_map : f_map

            # Preparation of the feature maps for to cost volumes
            curr_f_maps_lvl = vector_processing(tf.reshape(curr_f_maps_lvl, [b,h,w,nbre_cuts,-1]))
            curr_f_maps_lvl = tf.concat(tf.unstack(curr_f_maps_lvl, axis=3), axis=3)
            if prev_f_maps_time is not None:
                prev_f_maps_time = vector_processing(tf.reshape(prev_f_maps_time, [b,h,w,nbre_cuts,-1]))
                prev_f_maps_time = tf.concat(tf.unstack(prev_f_maps_time, axis=3), axis=3)

            # Manage level temporal memory
            if (not self.is_training) and prev_f_maps_time is None and prev_semantic_time is None:
                ## If it is a new sequence and we are at test time, assign predicted features and semantics of the previous sequence
                prev_semantic_time = self.semantic_prev_time
                prev_f_maps_time = self.prev_f_maps_time

            if prev_s_est_lvl is None:
                # Initial state of variables for the first level
                semantic_prev_lvl = tf.zeros([b, h, w, self.classes-1])
                other_class = tf.ones([b, h, w, 1])
                semantic_prev_lvl = tf.concat([semantic_prev_lvl, other_class], axis = 3)
                other_prev_lvl = tf.zeros([b, h, w, 4])
            else:
                ## Upscaling
                semantic_prev_lvl = tf.compat.v1.image.resize(prev_s_est_lvl["semantic"], [h, w], method = "nearest")
                other_prev_lvl = tf.compat.v1.image.resize(prev_s_est_lvl["other"], [h, w], method = "nearest")

            # Reinitialize temporal memory if sample is part of a new sequence
            # Note : sequences are supposed to be synchronized over the whole batch
            
            #### TESTING BLOCK ####
            ## Currently no difference in the two cases, but in the future there will be a difference
            if prev_semantic_time is None or new_traj[0]:
                
                with tf.name_scope("preprocessor"):

                    with tf.name_scope("input_prep"):
                        #max_d = 200.0
                        #input_features = [tf.clip_by_value(curr_depth_lvl, 0.001, max_d), semantic_prev_lvl]
                        input_features = [semantic_prev_lvl]
                        

                        if self.ablation.level_memory:
                            input_features.append(other_prev_lvl)
                        else:
                            print("Ignoring level memory")

                        if self.ablation.SNCV:
                            autocorr = cost_volume(curr_f_maps_lvl, curr_f_maps_lvl, 3, nbre_cuts=nbre_cuts)
                            input_features.append(autocorr)
                        else:
                            input_features.append(curr_f_maps_lvl)
                            print("Skipping SNCV")


                        f_input = tf.concat(input_features, axis=3)
                        

                with tf.name_scope("semantic_estimator"):
                    prev_out = self.semantic_refiner(f_input)
                    
                    semantic = prev_out[0][:, :, :, :self.classes]
                    other = prev_out[0][:, :, :, self.classes:]
                    
                    curr_s_est_lvl = {
                        "other": tf.identity(other),
                        "semantic": tf.identity(semantic),
                    }

                    if not self.is_training:
                        self.prev_f_maps_time.assign(curr_f_maps_lvl)
                        self.semantic_prev_time.assign(semantic)
                        
                        
                '''
                curr_semantic_time = tf.zeros(self.shape[:3] + [self.classes-1], dtype='float32')
                other_class = tf.ones(self.shape[:3] + [1], dtype='float32')
                curr_semantic_time = tf.concat([curr_semantic_time, other_class], axis = 3)
                
                if not self.is_training:
                    self.prev_f_maps_time.assign(curr_f_maps_lvl)
                    self.semantic_prev_time.assign(curr_semantic_time)
                    
                curr_s_est_lvl = {"semantic": semantic_prev_lvl, "other": other_prev_lvl}
                return curr_s_est_lvl
                '''
            else:
                
                with tf.name_scope("preprocessor"):

                    #para_prev_t = prev_d2para(prev_t_depth, rot, trans, camera)

                    #cv, para_prev_t_reproj = get_parallax_sweeping_cv(curr_f_maps, prev_f_maps, para_prev_t,
                                                                       #para_prev_l, rot, trans, camera, 4, nbre_cuts=nbre_cuts)

                    with tf.name_scope("input_prep"):
                        
                        #max_d = 200.0
                        #input_features = [tf.clip_by_value(curr_depth_lvl, 0.001, max_d), semantic_prev_lvl]
                        input_features = [semantic_prev_lvl]
                        

                        if self.ablation.level_memory:
                            input_features.append(other_prev_lvl)
                        else:
                            print("Ignoring level memory")

                        if self.ablation.SNCV:
                            autocorr = cost_volume(curr_f_maps_lvl, curr_f_maps_lvl, 3, nbre_cuts=nbre_cuts)
                            input_features.append(autocorr)
                        else:
                            input_features.append(curr_f_maps_lvl)
                            print("Skipping SNCV")

                        #if self.ablation.time_recurr:
                            #print("Doing nothing fo time recurrence ")
                            #input_features.append(tf.math.log(para_prev_t_reproj[:,:,:,4:5]*2**self.lvl_mul))
                        #else:
                            #print("Skipping time recurrence")

                        f_input = tf.concat(input_features, axis=3)
                        

                with tf.name_scope("semantic_estimator"):
                    prev_out = self.semantic_refiner(f_input)

                    #para = prev_out[0][:, :, :, :1]
                    semantic = prev_out[0][:, :, :, :self.classes]
                    other = prev_out[0][:, :, :, self.classes:]
                    

                    #para_curr_l = tf.exp(tf.clip_by_value(para, -7., 7.))/2**self.lvl_mul
                    #depth_prev_t = parallax2depth(para_curr_l, rot, trans, camera)
                    curr_s_est_lvl = {
                        "other": tf.identity(other),
                        "semantic": tf.identity(semantic),
                        #"depth": tf.identity(depth_prev_t),
                        #"parallax": tf.identity(para_curr_l),
                    }

                    if not self.is_training:
                        self.prev_f_maps_time.assign(curr_f_maps_lvl)
                        self.semantic_prev_time.assign(semantic)
                        
            #######################
            

            return curr_s_est_lvl

class SemanticEstimatorPyramid(ks.layers.Layer):
    # Decoder part of the architecture
    # Requires the feature map pyramid(s) produced by the encoder as input

    def __init__(self, settings, regularizer_weight=0.0004, trainable=True):
        super(SemanticEstimatorPyramid, self).__init__(trainable=trainable)
        # self.trainable = trainable
        self.levels = [
            SemanticEstimatorLevel(settings, i+1, regularizer_weight=regularizer_weight) for i in range(settings["nbre_lvls"])
        ]
        self.is_training = settings["is_training"]
        self.is_unsupervised = False #settings["unsupervised"]

    @tf.function
    def call(self, f_maps_pyrs, traj_samples, camera, training=False):

        s_est_seq = []
        for seq_i, (f_pyr_curr_time, sample) in enumerate(zip(f_maps_pyrs, traj_samples)):
            with tf.name_scope("SemanticEstimator_seq"):
                print("Seq sample %i" % seq_i)
                #print("DEPTH SAMPLE: ", sample['depth'])
                rot = sample['rot']
                trans = sample['trans']
                depth = sample['depth']
                h,w = depth.get_shape().as_list()[1:3]

                cnter = float(len(self.levels))
                s_est_curr_time = None

                # Loop over all the levels of the pyramid
                # Note : the deepest level has to be handled slightly differently due to the absence of deeper level
                for l, (f_maps_curr_lvl, level) in enumerate(zip(f_pyr_curr_time[::-1], self.levels[::-1])):
                    ## features and predicted semantics of previous time step
                    f_maps_prev_time = None
                    s_est_prev_time = None
                    if self.is_training and seq_i != 0:
                        f_maps_prev_time = f_maps_pyrs[seq_i - 1][-l - 1]
                        s_est_prev_time = s_est_seq[-1][-l - 1]["semantic"]
                        
                    print("f_maps_curr_lvl: ", f_maps_curr_lvl.get_shape())
                    local_camera = camera.copy()
                    local_camera["f"] /= 2. ** cnter
                    local_camera["c"] /= 2. ** cnter
                    #print("CNTER = ", cnter)
                    #print("F = ", local_camera["f"])
                    #print("C = ", local_camera["c"])
                    
                    #h1 = tf.cast(h/ (2 ** cnter), tf.uint8)
                    #w1 = tf.cast(w/ (2 ** cnter), tf.uint8)
                    
                    h1, w1 =  f_maps_curr_lvl.get_shape().as_list()[1:3]
                    print("h= ", h1)
                    print("w= ", w1)
                    depth_lvl = tf.image.resize(depth, [h1,w1])
                    print("LEVEL= ", l)
                    print("DEPTH LEVEL: ", depth_lvl)
                    
                    ## predicted semantics of previous level
                    if l != 0:
                        s_est_prev_lvl = s_est_curr_time[-1].copy()
                    else:
                        s_est_prev_lvl= None

                    local_rot = rot
                    local_trans = trans
                    new_traj = sample["new_traj"]

                    if s_est_curr_time is None:
                        s_est_curr_time = [level(f_maps_curr_lvl, None, depth_lvl, local_rot, local_trans, local_camera, new_traj,
                                            prev_f_maps_time=f_maps_prev_time, prev_semantic_time=s_est_prev_time)]
                    else:
                        s_est_curr_time.append(
                            level(f_maps_curr_lvl, s_est_prev_lvl, depth_lvl, local_rot, local_trans, local_camera, new_traj,
                                  prev_f_maps_time=f_maps_prev_time, prev_semantic_time=s_est_prev_time))
                    cnter -= 1.

                s_est_seq.append(s_est_curr_time[::-1])
        return s_est_seq


class M4Depth(ks.models.Model):
    """Tensorflow model of M4Depth"""

    def __init__(self, nbre_levels=6, is_training=False, ablation_settings=None, num_classes = 14, class_weights = None):
        super(M4Depth, self).__init__()

        if ablation_settings is None:
            self.ablation_settings = M4depthAblationParameters()
        else:
            self.ablation_settings = ablation_settings

        self.model_settings = {
            "nbre_lvls": nbre_levels,
            "is_training": is_training,
            "ablation" : self.ablation_settings,
            "classes": num_classes
        }

        #self.depth_type = depth_type

        self.encoder = FeaturePyramid(self.model_settings, regularizer_weight=0.)
        self.s_estimator = SemanticEstimatorPyramid(self.model_settings,
                                                 regularizer_weight=0.)

        self.step_counter = tf.Variable(initial_value=tf.zeros_initializer()(shape=[], dtype='int64'), trainable=False)
        self.summaries = []
        self.class_weights = class_weights

    @tf.function
    def call(self, data, training=False):
        traj_samples = data[0]
        camera = data[1]
        with tf.name_scope("M4Depth"):
            self.step_counter.assign_add(1)

            f_maps_pyrs = []
            for sample in traj_samples:
                f_maps_pyrs.append(self.encoder(sample['RGB_im']))

            s_maps_pyrs = self.s_estimator(f_maps_pyrs, traj_samples, camera, training)

            if training:
                return s_maps_pyrs
            else:
                h, w = traj_samples[-1]['RGB_im'].get_shape().as_list()[1:3]
                
                return {"semantic": tf.image.resize(s_maps_pyrs[-1][0]["semantic"], [h, w],
                                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)}
                #return {"depth": tf.image.resize(d_maps_pyrs[-1][0]["depth"], [h, w],
                                                 #method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)}

    @tf.function
    def train_step(self, data):
        with tf.name_scope("train_scope"):
            with tf.GradientTape() as tape:

                # Rearrange samples produced by the dataloader
                print("SAMPLE DATA SHAPE = ", data["semantic"].get_shape().as_list())
                seq_len = data["semantic"].get_shape().as_list()[1]
                traj_samples = [{} for i in range(seq_len)]
                attribute_list = ["semantic", "depth", "RGB_im", "new_traj", "rot", "trans"]
                #attribute_list = ["semantic", "RGB_im", "new_traj", "rot", "trans"]
                for key in attribute_list:
                    value_list = tf.unstack(data[key], axis=1)
                    for i, item in enumerate(value_list):
                        print("ITEM SHAPE = ", item.get_shape())
                        shape = item.get_shape()
                        traj_samples[i][key] = item

                gts = []
                for sample in traj_samples:
                    #gts.append({"depth":sample["depth"], "parallax": depth2parallax(sample["depth"], sample["rot"], sample["trans"], data["camera"])})
                    gts.append({"semantic":sample["semantic"]})
                preds = self([traj_samples, data["camera"]], training=True)
                ## Yara: The defined m4depth loss (different from catgorical crossentropy loss)
                loss = self.m4depth_loss(gts, preds)
                #print("LOSS = ", loss.numpy())

            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            # Update metrics (includes the metric that tracks the loss)

        with tf.name_scope("summaries"):
            #max_d = 200.
            #gt_d_clipped = tf.clip_by_value(traj_samples[-1]['depth'], 1., max_d)
            gt_s = traj_samples[-1]['semantic']
            tf.summary.image("RGB_im", traj_samples[-1]['RGB_im'], step=self.step_counter)
            #im_reproj, _ = reproject(traj_samples[-2]['RGB_im'], traj_samples[-1]['depth'],
                                     #traj_samples[-1]['rot'], traj_samples[-1]['trans'], data["camera"])
            #tf.summary.image("camera_prev_t_reproj", im_reproj, step=self.step_counter)

            #tf.summary.image("depth_gt", tf.math.log(gt_d_clipped) / tf.math.log(max_d), step=self.step_counter)
            tf.summary.image("semantic_gt", gt_s, step=self.step_counter)
            #print("PREDS= ", preds)
            #print("LAST PREDS= ", preds[-1])
            for i, est in enumerate(preds[-1]):
                #print("EST= ", est)
                #d_est_clipped = tf.clip_by_value(est["depth"], 1., max_d)
                s_est = tf.expand_dims(tf.math.argmax(est["semantic"], -1), -1)
                self.summaries.append([tf.summary.image, "semantic_lvl_%i" % i, s_est])
                tf.summary.image("semantic_lvl_%i" % i, s_est, step=self.step_counter)
                #self.summaries.append(
                    #[tf.summary.image, "depth_lvl_%i" % i, tf.math.log(d_est_clipped) / tf.math.log(max_d)])
                #tf.summary.image("depth_lvl_%i" % i, tf.math.log(d_est_clipped) / tf.math.log(max_d),
                                 #step=self.step_counter)

        with tf.name_scope("metrics"):
            #gt = gts[-1]["depth"]
            gt = gts[-1]["semantic"]
            s_pred = preds[-1][0]["semantic"]
            #s_est = tf.expand_dims(tf.math.argmax(s_pred, -1), -1)
            est = tf.image.resize(s_pred, gt.get_shape()[1:3],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            #est = tf.image.resize(preds[-1][0]["depth"], gt.get_shape()[1:3],
                                  #method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            #max_d = 80.
            #gt = tf.clip_by_value(gt, 0.00, max_d)
            #est = tf.clip_by_value(est, 0.001, max_d)
            ## sparse categorical crossentropy loss
            #print("GT = ", gt)
            #print("PRED = ", est)
            est_argmax = tf.math.argmax(est, 3)
            self.compiled_metrics.update_state(gt, est_argmax)
            out_dict = {m.name: m.result() for m in self.metrics}
            out_dict["loss"] = loss

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return out_dict

    @tf.function
    def test_step(self, data):
        # expects one sequence element at a time (batch dim required and is free to set)"
        data_format = len(data["semantic"].get_shape().as_list())

        # If sequence was received as input, compute performance metrics only on its last frame (required for KITTI benchmark))
        if data_format == 5:
            seq_len = data["semantic"].get_shape().as_list()[1]
            traj_samples = [{} for i in range(seq_len)]
            # attribute_list = ["semantic", "RGB_im", "new_traj", "rot", "trans"]
            attribute_list = ["semantic", "depth", "RGB_im", "new_traj", "rot", "trans"]
            for key in attribute_list:
                value_list = tf.unstack(data[key], axis=1)
                for i, item in enumerate(value_list):
                    shape = item.get_shape()
                    traj_samples[i][key] = item

            gts = []
            for sample in traj_samples:
                gts.append({"semantic":sample["semantic"]})
            preds = self([traj_samples, data["camera"]], training=False)
            gt = data["semantic"][:,-1,:,:,:]
            est = preds["semantic"]
            new_traj=False
        else:
            preds = self([[data], data["camera"]], training=False)
            gt = data["semantic"]
            est = preds["semantic"]
            new_traj = data["new_traj"]

        with tf.name_scope("metrics"):
            # Compute performance scores

            #max_d = 80.
            #gt = tf.clip_by_value(gt, 0.0, max_d) 
            #est = tf.clip_by_value(est, 0.001, max_d)

            if not new_traj:
                est_argmax = tf.math.argmax(est, 3)
                self.compiled_metrics.update_state(gt, est_argmax)

        # Return a dict mapping metric names to current value.
        out_dict = {m.name: m.result() for m in self.metrics}
        return out_dict

    @tf.function
    def predict_step(self, data):
        # expects one sequence element at a time (batch dim is required and is free to be set)"
        preds = self([[data], data["camera"]], training=False)

        with tf.name_scope("metrics"):
            est = preds["semantic"]
            s_est = tf.expand_dims(tf.math.argmax(est, -1), -1)
            return_data = {
                "image": data["RGB_im"],
                "semantic": s_est,
                "new_traj": data["new_traj"]
            }
        return return_data

    @tf.function
    def m4depth_loss(self, gts, preds):
        with tf.name_scope("loss_function"):
            scce = tf.keras.losses.SparseCategoricalCrossentropy()
            cce = tf.keras.losses.CategoricalCrossentropy()
            fcce = SparseCategoricalFocalLoss(gamma=2)
            
            loss = 0.
            
            #for gt, pred_pyr in zip(gts[1:], preds[1:]):  # Iterate over sequence
            for gt, pred_pyr in zip(gts[0:], preds[0:]):  # Iterate over sequence
                
                nbre_points = 0.
                
                gt_semantic = gt["semantic"]

                
                for i, pred in enumerate(pred_pyr):  # Iterate over the outputs produced by the different levels
                    #pred_depth = preprocess(pred["depth"])
                    pred_semantic = pred["semantic"]

                    # Compute loss term
                    b, h, w = pred_semantic.get_shape().as_list()[:3]
                    nbre_points += h * w
                    gt_resized = tf.image.resize(gt_semantic, [h, w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                    #print("GT = ", gt_resized)
                    #print("GT SHAPE = ", gt_resized.get_shape().as_list())
                    #print("PREDICTION SHAPE = ", pred_semantic.get_shape().as_list())
                    #print("PREDICTION = ", pred_semantic)
                    #loss_term = scce(gt_resized, pred_semantic)
                    
                    #loss_focal = fcce(gt_resized, pred_semantic)
                    #print("LOSS TERM: ", loss_term)
                    
                    gt_one_hot = tf.one_hot(gt_resized, depth = self.model_settings["classes"])
                    gt_one_hot = tf.squeeze(gt_one_hot, axis = 3)
                    gt_one_hot = gt_one_hot*[4.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 6.0, 4.0, 4.0, 6.0]
                    loss_term = cce(gt_one_hot, pred_semantic)
                    #loss_dice = dice_coef_loss(gt_one_hot, pred_semantic, M=12)
                    
                    #print("LOSS DICE: ", loss_dice)
                    
                    #loss_term = scce(gt_resized, pred_semantic).numpy()
                    
                    #loss_term = tf.reduce_mean(gt_resized - pred_depth)
                    loss += loss_term / (float(len(gts)))
                    #loss += loss_focal / (float(len(gts)))
                    #loss += loss_dice / (float(len(gts)))
                    
                    #print("LOSS DICE = ", loss_dice)
                    #print("LOSS CATEGORICAL = ", loss_term)
                    
                
            return loss
