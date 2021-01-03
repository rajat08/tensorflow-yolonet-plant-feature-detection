import os
import numpy as np
import cv2
import pickle
import config as cfg

class image_voc(object):
    def __init__(self, phase, rebuild=False):
        self.data_path = cfg.DATA_PATH
        self.label_path = cfg.LABEL_PATH
        self.cache_path = cfg.CACHE_PATH
        self.batch_size = cfg.BATCH_SIZE
        self.image_size = cfg.IMAGE_SIZE
        self.cell_num = cfg.CELL_NUM
        self.cell_size = self.image_size / self.cell_num
        self.classes = cfg.CLASSES
        self.class_to_ind = dict(zip(self.classes, range(len(self.classes))))
        self.phase = phase
        self.rebuild = rebuild
        self.cursor = 0   # the "pointer" used to access label items
        self.epoch = 1
        self.gt_labels = None
        #prepare the labels
        self.prepare()

    def get(self):
        images = np.zeros(
            (self.batch_size, self.image_size, self.image_size, 3)) # 3 is the number of channels in the image
        labels = np.zeros(
            (self.batch_size, self.cell_num, self.cell_num, 23)) # 6 is the number of elements in the label: Pr_object,x,y,Pr_bud, Pr_flower, Pr_fruit + 17 empty label
        count = 0
        while count < self.batch_size:
            imname = self.gt_labels[self.cursor]['imname']
            images[count, :, :, :] = self.image_read(imname)
            labels[count, :, :, :] = self.gt_labels[self.cursor]['label']
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.gt_labels):
                np.random.shuffle(self.gt_labels)
                self.cursor = 0
                self.epoch += 1
        return images, labels

    def image_read(self, imname):
        image = cv2.imread(imname)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0
        return image

    def prepare(self):
        gt_labels = self.load_labels()
        #shuffle the data
        np.random.shuffle(gt_labels)
        self.gt_labels = gt_labels
        return gt_labels

    def process_line(self, line):
        label = np.zeros((self.cell_num, self.cell_num, 23))
        # parse the line
        # get the specie name 
        species = line[0].split(".")[0]
        image_path = "images/" + species + "/"+line[0]
        # input image
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # compute the ratio of the length and with of the shrunk image over original image
        h_ratio = 1.0 * self.image_size / img.shape[0]
        w_ratio = 1.0 * self.image_size / img.shape[1]
        # coordinate lists
        bud = line[1].split(";")
        flower = line[2].split(";")
        fruit = line[3].split("\n")[0].split(";")
        # construct the ground truth
        # x,y are converted to the
        for item in bud:
            if item != " ":
                x = float(item.split("_")[0]) * w_ratio
                y = float(item.split("_")[1]) * h_ratio
                # find the cell that this point belongs to
                x_cell = int(np.floor(x / self.cell_size))
                y_cell = int(np.floor(y / self.cell_size))
                # if the coordinates go out of bound, do not use this 
                if x_cell >= self.cell_num or y_cell >= self.cell_num:
                    continue
                # determine whether this cell has been occupied i.e. probability of existing object == 1
                # unassigned cell
                if label[x_cell,y_cell,0] == 0:
                    label[x_cell, y_cell, 0] = 1
                    label[x_cell, y_cell, 1] = x
                    label[x_cell, y_cell, 2] = y
                    # this is a bud
                    label[x_cell, y_cell, 3] = 1
        # draw flower
        for item in flower:
            if item != " ":
                x = float(item.split("_")[0]) * w_ratio
                y = float(item.split("_")[1]) * h_ratio
                # find the cell that this point belongs to
                x_cell = int(np.floor(x / self.cell_size))
                y_cell = int(np.floor(y / self.cell_size))
                # if the coordinates go out of bound, do not use this 
                if x_cell >= self.cell_num or y_cell >= self.cell_num:
                    continue
                # determine whether this cell has been occupied i.e. probability of existing object == 1
                # unassigned cell
                if label[x_cell,y_cell,0] == 0:
                    label[x_cell, y_cell, 0] = 1
                    label[x_cell, y_cell, 1] = x
                    label[x_cell, y_cell, 2] = y
                    # this is a flower
                    label[x_cell, y_cell, 4] = 1
        # draw flower
        for item in fruit:
            if item != " ":
                x = float(item.split("_")[0]) * w_ratio
                y = float (item.split("_")[1]) * h_ratio
                # find the cell that this point belongs to
                x_cell = int(np.floor(x / self.cell_size))
                y_cell = int(np.floor(y / self.cell_size))
                # if the coordinates go out of bound, do not use this 
                if x_cell >= self.cell_num or y_cell >= self.cell_num:
                    continue
                # determine whether this cell has been occupied i.e. probability of existing object == 1
                # unassigned cell
                if label[x_cell, y_cell, 0] == 0:
                    label[x_cell, y_cell, 0] = 1
                    label[x_cell, y_cell, 1] = x
                    label[x_cell, y_cell, 2] = y
                    # this is a fruit
                    label[x_cell, y_cell, 5] = 1
        return {'imname': image_path, 'label': label}
 
    def load_labels(self):
        # whether we want to rebuild the label
        cache_file = os.path.join(
            self.cache_path, 'herb_' + self.phase + '_gt_labels.pkl')

        if os.path.isfile(cache_file) and not self.rebuild:
            print('Loading gt_labels from: ' + cache_file)
            with open(cache_file, 'rb') as f:
                gt_labels = pickle.load(f)
            return gt_labels
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        gt_labels = []
        print('Processing gt_labels from: ' + self.data_path)
        with open(self.label_path, "r") as ground_truth:
            line = ground_truth.readline()
            while line:
                line = line.split(",")
                processed_line = self.process_line(line)
                gt_labels.append(processed_line)
                line = ground_truth.readline()
        return gt_labels
