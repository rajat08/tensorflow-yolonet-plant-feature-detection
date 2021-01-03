import numpy as np
import tensorflow as tf
import config as cfg

slim = tf.contrib.slim


class YOLONet(object):

    def __init__(self, is_training=True):
        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.cell_num = cfg.CELL_NUM
        self.dist_threshold = cfg.DIST_THRESHOLD
        self.centers_per_cell = cfg.CENTERS_PER_CELL
        # The format of the output: it predicts x,y and probability being an object(3 parameters) and the conditional
        # probability of being a particular class provided being an object(3 parameters here).
        self.output_size = (self.cell_num * self.cell_num) * \
                           (self.num_class + self.centers_per_cell * 5)
        self.scale = 1.0 * self.image_size / self.cell_num

        self.boundary1 = self.cell_num * self.cell_num * self.num_class
        self.boundary2 = self.boundary1 + \
                         self.cell_num * self.cell_num * self.centers_per_cell

        self.object_scale = cfg.OBJECT_SCALE
        self.noobject_scale = cfg.NOOBJECT_SCALE
        self.class_scale = cfg.CLASS_SCALE
        self.coord_scale = cfg.COORD_SCALE

        self.learning_rate = cfg.LEARNING_RATE
        self.batch_size = cfg.BATCH_SIZE
        self.alpha = cfg.ALPHA

        self.offset = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_num)] * self.cell_num * 1),
            (1, self.cell_num, self.cell_num)), (1, 2, 0))

        self.images = tf.placeholder(
            tf.float32, [None, self.image_size, self.image_size, 3],
            name='images')
        self.logits = self.build_network(
            self.images, num_outputs=self.output_size, alpha=self.alpha,
            is_training=is_training)

        # only compute loss and update parameters
        if is_training:
            self.labels = tf.placeholder(
                tf.float32,
                [None, self.cell_num, self.cell_num, 3 + self.num_class])
            self.loss_layer(self.logits, self.labels)
            self.total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('total_loss', self.total_loss)


    def build_network(self,
                      images,
                      num_outputs,
                      alpha,
                      keep_prob=0.5,  #probability for keeping
                      is_training=True,
                      scope='yolo'):
        with tf.variable_scope(scope):
            with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                activation_fn=leaky_relu(alpha),
                weights_regularizer=slim.l2_regularizer(0.0005),
                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)
            ):
                net = tf.pad(
                    images, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]),
                    name='pad_1')
                
                net = slim.conv2d(
                    net, 64, 7, 2, padding='VALID', scope='conv_2')
                
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_3')
                net = slim.conv2d(net, 192, 3, scope='conv_4')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5')
                net = slim.conv2d(net, 128, 1, scope='conv_6')
                net = slim.conv2d(net, 256, 3, scope='conv_7')
                net = slim.conv2d(net, 256, 1, scope='conv_8')
                net = slim.conv2d(net, 512, 3, scope='conv_9')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10')
                net = slim.conv2d(net, 256, 1, scope='conv_11')
                net = slim.conv2d(net, 512, 3, scope='conv_12')
                net = slim.conv2d(net, 256, 1, scope='conv_13')
                net = slim.conv2d(net, 512, 3, scope='conv_14')
                net = slim.conv2d(net, 256, 1, scope='conv_15')
                net = slim.conv2d(net, 512, 3, scope='conv_16')
                net = slim.conv2d(net, 256, 1, scope='conv_17')
                net = slim.conv2d(net, 512, 3, scope='conv_18')
                net = slim.conv2d(net, 512, 1, scope='conv_19')
                net = slim.conv2d(net, 1024, 3, scope='conv_20')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21')
                net = slim.conv2d(net, 512, 1, scope='conv_22')
                net = slim.conv2d(net, 1024, 3, scope='conv_23')
                net = slim.conv2d(net, 512, 1, scope='conv_24')
                net = slim.conv2d(net, 1024, 3, scope='conv_25')
                net = slim.conv2d(net, 1024, 3, scope='conv_26')
                net = tf.pad(
                    net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]),
                    name='pad_27')
                net = slim.conv2d(
                    net, 1024, 3, 2, padding='VALID', scope='conv_28')
                net = slim.conv2d(net, 1024, 3, scope='conv_29')
                net = slim.conv2d(net, 1024, 3, scope='conv_30')
                net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
                net = slim.flatten(net, scope='flat_32')
                net = slim.fully_connected(net, 512, scope='fc_33')
                net = slim.fully_connected(net, 4096, scope='fc_34')
                net = slim.dropout(
                    net, keep_prob=keep_prob, is_training=is_training,
                    scope='dropout_35')

                net = slim.fully_connected(
                    net, num_outputs, activation_fn=None, scope='fc_36')
        return net

    def calc_dist(self, centers1, centers2, scope='iou'):
        """calculate the distance evaluation of two centers
           It is computed by 1 - max({sqrt[(x_1 - x_2)^2 + (y_1 - y_2)^2] / 2 * cell_length}, 1}
           i.e. The closer they are, the more confident the prediction is
        Args:
          centers1: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, CENTERS_PER_CELL, 2]  ====> (x_center, y_center)
          centers2: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, CENTERS_PER_CELL, 2] ===> (x_center, y_center)
        Return:
          iou: 4-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, CENTERS_PER_CELL]
        """
        with tf.variable_scope(scope):
            distance = tf.sqrt((centers1[..., 0] - centers2[..., 0]) ** 2 + (centers1[..., 1] - centers2[..., 1]) ** 2)
            cell_length = self.image_size/self.cell_num
            distance = distance / (cell_length)
            distance = tf.maximum(distance, 1.0)
        return tf.clip_by_value(1 - distance, 0.0, 1.0)

   
    def loss_layer(self, predicts, labels, scope='loss_layer'):

        with tf.variable_scope(scope):
            # predicted conditional probabilities
            predict_classes = tf.reshape(
                predicts[:, :self.boundary1],
                [self.batch_size, self.cell_num, self.cell_num, self.num_class])
            # predicted object probability
            predict_scales = tf.reshape(
                predicts[:, self.boundary1:self.boundary2],
                [self.batch_size, self.cell_num, self.cell_num, self.centers_per_cell])
            predict_scales = predict_scales[:,:,:,0:1]
            # predicted coordinate
            predict_centers = tf.reshape(
                predicts[:, self.boundary2:],
                [self.batch_size, self.cell_num, self.cell_num, self.centers_per_cell, 4])
            predict_centers = predict_centers[:,:,:,0:1,0:2]
            # ground truth for object probability
            response = tf.reshape(
                labels[..., 0],
                [self.batch_size, self.cell_num, self.cell_num, 1])
            # center coordinates of objects
            centers = tf.reshape(
                labels[..., 1:3],
                [self.batch_size, self.cell_num, self.cell_num, 1, 2])
            centers = tf.tile(
                centers, [1, 1, 1, 1, 1]) / self.image_size
            #conditional probability for each class
            classes = labels[..., 3:]

            offset = tf.reshape(
                tf.constant(self.offset, dtype=tf.float32),
                [1, self.cell_num, self.cell_num, 1])
            offset = tf.tile(offset, [self.batch_size, 1, 1, 1])
            offset_tran = tf.transpose(offset, (0, 2, 1, 3))
            predict_centers_tran = tf.stack(
                [(predict_centers[..., 0] + offset) / self.cell_num,
                 (predict_centers[..., 1] + offset_tran) / self.cell_num], axis=-1)

            dist_predict_truth = self.calc_dist(predict_centers_tran, centers)

            # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            # find the centers with the highest probability within the centers
            best_dist = tf.reduce_max(dist_predict_truth, 3, keepdims=True)
            object_mask = tf.cast(
                (best_dist > self.dist_threshold), tf.float32)

            # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            noobject_mask = tf.ones_like(
                object_mask, dtype=tf.float32) - object_mask

            centers_tran = tf.stack(
                [centers[..., 0] * self.cell_num - offset,
                 centers[..., 1] * self.cell_num - offset_tran], axis=-1)

            # class_loss
            class_delta = response * (predict_classes - classes)
            class_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]),
                name='class_loss') * self.class_scale

            # object_loss
            object_delta = response * (predict_scales - best_dist)
            object_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),
                name='object_loss') * self.object_scale

            # noobject_loss
            noobject_delta = predict_scales * (tf.ones_like(response, dtype=tf.float32) - response)
            noobject_loss = tf.reduce_mean(
                tf.reduce_sum( noobject_mask * tf.square(noobject_delta), axis=[1, 2, 3]),
                name='noobject_loss') * self.noobject_scale

            # coord_loss
            coord_mask = tf.expand_dims(response, 4)
            centers_delta = coord_mask * (predict_centers - centers_tran)
            coord_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(centers_delta), axis=[1, 2, 3, 4]),
                name='coord_loss') * self.coord_scale

            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(noobject_loss)
            tf.losses.add_loss(coord_loss)

            tf.summary.scalar('class_loss', class_loss)
            tf.summary.scalar('object_loss', object_loss)
            tf.summary.scalar('noobject_loss', noobject_loss)
            tf.summary.scalar('coord_loss', coord_loss)

            tf.summary.histogram('centers_delta_x', centers_delta[..., 0])
            tf.summary.histogram('centers_delta_y', centers_delta[..., 1])

def leaky_relu(alpha):
    def op(inputs):
        return tf.nn.leaky_relu(inputs, alpha=alpha, name='leaky_relu')
    return op
