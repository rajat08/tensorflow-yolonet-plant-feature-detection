import os
import argparse
import datetime
import tensorflow as tf
import numpy as np
import config as cfg
from YOLO_network import YOLONet
from timer import Timer
from image_voc import image_voc

'''
def get_n_cores():  
    nslots = os.getenv('NSLOTS')
    if nslots is not None:
        return int(nslots)
    raise ValueError('Environment variable NSLOTS is not defined.')
'''
slim = tf.contrib.slim
class Solver(object):

    def __init__(self, net, data):
        self.net = net 
        self.data = data
        self.weights_file = cfg.WEIGHTS_FILE
        self.max_iter = cfg.MAX_ITER
        self.initial_learning_rate = cfg.LEARNING_RATE
        self.decay_steps = cfg.DECAY_STEPS
        self.decay_rate = cfg.DECAY_RATE
        self.staircase = cfg.STAIRCASE
        self.summary_iter = cfg.SUMMARY_ITER
        self.save_iter = cfg.SAVE_ITER

        self.output_dir = os.path.abspath(os.path.join(
            cfg.OUTPUT_DIR, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.save_cfg()

        self.variable_to_restore = tf.global_variables()
        self.saver = tf.train.Saver(self.variable_to_restore, max_to_keep=None)
        self.ckpt_file = os.path.join(self.output_dir, 'yolo.ckpt')
        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.output_dir, flush_secs=60)

        self.global_step = tf.train.create_global_step()
        self.learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate, self.global_step, self.decay_steps,
            self.decay_rate, self.staircase, name='learning_rate')
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate)
        self.train_op = slim.learning.create_train_op(
            self.net.total_loss, self.optimizer, global_step=self.global_step)

        # turn on the GPU
        config = tf.ConfigProto(
            intra_op_parallelism_threads= 1, #get_n_cores()-1,
            inter_op_parallelism_threads=1,
            allow_soft_placement=True, 
            log_device_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        # variable initialization
        self.sess.run(tf.global_variables_initializer())
        
        if self.weights_file is not None:
            # convert to absolute path
            abs_path = os.path.abspath(self.weights_file)
            print('Restoring weights from: ' + abs_path)
            self.saver.restore(self.sess, abs_path)

        self.writer.add_graph(self.sess.graph)
    
    def train(self):

        train_timer = Timer()
        load_timer = Timer()

        for step in range(1, self.max_iter + 1):
            print('training iteration:' + str(step))
            load_timer.tic()
            images, labels = self.data.get()
            load_timer.toc()
            feed_dict = {self.net.images: images,
                         self.net.labels: labels}
            if step % self.summary_iter == 0:
                if step % (100) == 0:

                    train_timer.tic()
                    print('session is running...')
                    summary_str, loss, _ = self.sess.run(
                        [self.summary_op, self.net.total_loss, self.train_op],
                        feed_dict=feed_dict)
                    print('session is finished.')
                    train_timer.toc()

                    log_str = "{} Epoch: {}, Step: {}, Learning rate: {}, Loss: {:5.3f}\nSpeed: {:.3f}s/iter,Load: {:.3f}s/iter, Remain: {}".format(
                        datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                        self.data.epoch,
                        int(step),
                        round(self.learning_rate.eval(session=self.sess), 6),
                        loss,
                        train_timer.average_time,
                        load_timer.average_time,
                        train_timer.remain(step, self.max_iter))
                    print(log_str)

                else:
                    train_timer.tic()
                    print('session is running...')
                    summary_str, _ = self.sess.run(
                        [self.summary_op, self.train_op],
                        feed_dict=feed_dict)
                    print('session is finished.')
                    train_timer.toc()

                self.writer.add_summary(summary_str, step)

            else:
                train_timer.tic()
                print('session is running...')
                self.sess.run(self.train_op, feed_dict=feed_dict)
                print('session is finished.')
                train_timer.toc()

            if step % self.save_iter == 0:
                print('{} Saving checkpoint file to: {}'.format(
                    datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                    self.output_dir))
                self.saver.save(
                    self.sess, self.ckpt_file, global_step=self.global_step)   #filename contains 
                    
    def save_cfg(self):
        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:
            cfg_dict = cfg.__dict__
            for key in sorted(cfg_dict.keys()):
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)

def update_config_paths(data_dir, weights_file):
    cfg.DATA_PATH = data_dir
    cfg.CACHE_PATH = os.path.join(cfg.NETWORK_PATH, 'cache')
    cfg.OUTPUT_DIR = os.path.join(cfg.NETWORK_PATH, 'output')
    cfg.WEIGHTS_DIR = os.path.join(cfg.NETWORK_PATH, 'weights')
    cfg.WEIGHTS_FILE = os.path.join(cfg.WEIGHTS_DIR, weights_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='weights/YOLO_small.ckpt', type=str)
    parser.add_argument('--data_dir', default="training", type=str)
    parser.add_argument('--threshold', default=0.2, type=float)
    parser.add_argument('--iou_threshold', default=0.5, type=float)
    parser.add_argument('--gpu', default="0", type=str)   #
    args = parser.parse_args()


    # try to restore the training from checkpoint file
    if args.weights is not None:
        cfg.WEIGHTS_FILE  = os.path.join(cfg.OUTPUT_DIR, args.weights)

    if args.gpu is not None:
        print("training with GPU id: " + args.gpu)
        cfg.GPU = args.gpu

    if args.data_dir != cfg.DATA_PATH:
        update_config_paths(args.data_dir, args.weights)

    #os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

    yolo = YOLONet()
    images = image_voc('train')

    solver = Solver(yolo, images)

    print('Start training ...')
    solver.train()
    print('Done training.')

if __name__ == '__main__':
    main()
