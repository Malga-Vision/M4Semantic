import tensorflow as tf
from .generic1 import *

class DataLoaderAeroscapes(DataLoaderGeneric):
    """Dataloader for the Aeroscapes dataset
    """
    def __init__(self, out_size=[720, 1280], crop=False):
        super(DataLoaderAeroscapes, self).__init__('aeroscapes')

        self.in_size = [720, 1280]
        self.depth_type = "map"
        self.class_count = 12
        self.class_index = {
        0: [(0, 0, 0), 'Background'],
        1: [(192, 128, 128),'Person'],
        2: [(0, 128, 0),  'Bike'],
        3: [(128, 128, 128),  'Car'],
        4: [(128, 0, 0), 'Drone'],   
        5: [(0, 0, 128),   'Boat'],
        6: [(192, 0, 128),  'Animal'],
        7: [(192, 0, 0),  'Obstacle'],
        8: [(192, 128, 0),  'Construction'],
        9: [(0, 64, 0),  'Vegetation'],
        10: [(128, 128, 0),  'Road'],
        11: [(0, 128, 128),  'Sky']
        }
        self.class_weight = [4.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 6.0, 4.0, 4.0, 6.0]

    def _set_output_size(self, out_size=[720, 1280]):
        self.out_size = out_size
        self.long_edge = 0 if out_size[0]>=out_size[1] else 1
        if self.crop:
            self.intermediate_size = [out_size[self.long_edge], out_size[self.long_edge]]
        else:
            self.intermediate_size = out_size
        self.fx = 0.5 * self.intermediate_size[1]
        self.fy = 0.5 * self.intermediate_size[0]
        self.cx = 0.5 * self.intermediate_size[1]
        self.cy = 0.5 * self.intermediate_size[0]

    def get_dataset(self, usecase, settings, batch_size=3, out_size=[720, 1280], crop=False):
        self.crop = crop
        if (usecase == "eval" or usecase=="predict") and self.crop:
            return AttributeError("Crop option should be disabled when evaluating or predicting samples")
        super(DataLoaderAeroscapes, self).get_dataset(usecase, settings, batch_size=batch_size, out_size=out_size)

    @tf.function
    def _decode_samples(self, data_sample):
        file = tf.io.read_file(tf.strings.join([self.db_path, data_sample['camera_l']], separator='/'))
        image = tf.io.decode_jpeg(file, channels=3)
        rgb_image = tf.cast(image, dtype=tf.float32)/255.

        camera_data = {
            "f": tf.convert_to_tensor([self.fx, self.fy]),
            "c": tf.convert_to_tensor([self.cx, self.cy]),
        }
        out_data = {}
        out_data["camera"] = camera_data.copy()
        out_data['RGB_im'] = tf.reshape(tf.image.resize(rgb_image, self.intermediate_size), self.intermediate_size+[3])
        out_data['rot'] = tf.cast(tf.stack([data_sample['qw'],data_sample['qx'],data_sample['qy'],data_sample['qz']], 0), dtype=tf.float32)
        out_data['trans'] = tf.cast(tf.stack([data_sample['tx'],data_sample['ty'],data_sample['tz']], 0), dtype=tf.float32)
        out_data['new_traj'] = tf.math.equal(data_sample['id'], 0)
        
        # Load semantic data only if they are available
        if 'semantic' in data_sample:
           file = tf.io.read_file(tf.strings.join([self.db_path, data_sample['semantic']], separator='/'))
           image = tf.image.decode_png(file)
           semantic = tf.reshape(image[:,:,0], self.in_size + [1])
           semantic = tf.image.resize(semantic, self.intermediate_size, method = "nearest")
           out_data['semantic'] = tf.image.resize(semantic, self.intermediate_size, method = "nearest")
        # Load depth data only if they are available
        if 'depth' in data_sample:
            file = tf.io.read_file(tf.strings.join([self.db_path, data_sample['depth']], separator='/'))
            image = tf.io.decode_raw(file, tf.float32)
            image = image[-(self.in_size[0]*self.in_size[1]):]
            depth = tf.reshape(tf.cast(image, dtype=tf.float32), self.in_size+[1])
            # WARNING we disable areas with no color information
            out_data['depth'] = tf.reshape(tf.image.resize(depth, self.out_size, method='nearest'), self.out_size+[1])
        return out_data

    def _perform_augmentation(self):
        # flip and transpose image

        if not self.usecase == "finetune":
            self._augmentation_step_flip()

            # we can transpose h and w dimensions if images have a square shape as a data augmentation
            if self.intermediate_size[0] == self.intermediate_size[1]:
                im_col = self.out_data["RGB_im"]
                im_depth = self.out_data["depth"]
                im_semantic = self.out_data["semantic"]
                rot = self.out_data["rot"]
                trans = self.out_data["trans"]

                def do_nothing():
                    #return [im_col, im_semantic, rot, trans]
                    return [im_col, im_semantic, im_depth, rot, trans]

                def true_transpose():
                    col = tf.transpose(im_col, perm=[0, 2, 1, 3])
                    dep = tf.transpose(im_depth, perm=[0, 2, 1, 3])
                    semantic = tf.transpose(im_semantic, perm=[0, 2, 1, 3])
                    r = tf.stack([rot[:, 0], -rot[:, 2], -rot[:, 1], -rot[:, 3]], axis=1)
                    t = tf.stack([trans[:, 1], trans[:, 0], trans[:, 2]], axis=1)
                    #return [col, semantic, r, t]
                    return [col, semantic, dep, r, t]

                p_order = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
                pred = tf.less(p_order, 0.5)
                im_col, im_semantic, im_depth, rot, trans = tf.cond(pred, true_transpose, do_nothing)
                
                self.out_data["depth"] = im_depth
                self.out_data["semantic"] = im_semantic
                self.out_data["RGB_im"] = im_col
                self.out_data["rot"] = rot
                self.out_data["trans"] = trans

        # crop image to the desired output size
        if self.crop:
            if self.long_edge == 0:
                diff = self.intermediate_size[1]-self.out_size[1]
                offset = tf.random.uniform(shape=[], minval=0, maxval=diff, dtype=tf.int32)
                self.out_data['RGB_im'] = tf.slice(self.out_data['RGB_im'], [0, 0, offset, 0], [self.seq_len, self.out_size[0], self.out_size[1], 3])
                self.out_data['depth'] = tf.slice(self.out_data['depth'], [0, 0, offset, 0], [self.seq_len, self.out_size[0], self.out_size[1], 1])
                self.out_data['semantic'] = tf.slice(self.out_data['semantic'], [0, 0, offset, 0], [self.seq_len, self.out_size[0], self.out_size[1], 1])
                self.out_data['camera']['c'] = tf.convert_to_tensor([self.out_data['camera']['c'][0]-tf.cast(offset, tf.float32), self.out_data['camera']['c'][1]])
            else:
                diff = self.intermediate_size[0]-self.out_size[0]
                offset = tf.random.uniform(shape=[], minval=0, maxval=diff, dtype=tf.int32)
                self.out_data['RGB_im'] = tf.slice(self.out_data['RGB_im'], [0, offset, 0, 0], [self.seq_len, self.out_size[0],  self.out_size[1], 3])
                self.out_data['depth'] = tf.slice(self.out_data['depth'], [0, offset, 0, 0], [self.seq_len, self.out_size[0], self.out_size[1], 1])
                self.out_data['semantic'] = tf.slice(self.out_data['semantic'], [0, offset, 0, 0], [self.seq_len, self.out_size[0], self.out_size[1], 1])
                self.out_data['camera']['c'] = tf.convert_to_tensor([self.out_data['camera']['c'][0], self.out_data['camera']['c'][1]-tf.cast(offset, tf.float32)])
            self.out_data['RGB_im'] = tf.reshape(self.out_data['RGB_im'], [self.seq_len, self.out_size[0],  self.out_size[1], 3])
            self.out_data['depth'] = tf.reshape(self.out_data['depth'], [self.seq_len, self.out_size[0],  self.out_size[1], 1])
            self.out_data['semantic'] = tf.reshape(self.out_data['semantic'], [self.seq_len, self.out_size[0],  self.out_size[1], 1])

        self._augmentation_step_color()
