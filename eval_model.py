import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
import config as cfg
from YOLO_network import YOLONet
from timer import Timer

SAVE_IMG = True
class Detector(object):

    def __init__(self, net, weight_file):
        self.net = net
        self.weights_file = weight_file

        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.cell_num = cfg.CELL_NUM
        self.cell_size = self.image_size / self.cell_num
        self.centers_per_cell = cfg.CENTERS_PER_CELL
        self.threshold = cfg.THRESHOLD
        self.dist_threshold = cfg.DIST_THRESHOLD
        self.boundary1 = self.cell_num * self.cell_num * self.num_class
        self.boundary2 = self.boundary1 + \
                         self.cell_num * self.cell_num * self.centers_per_cell

        # this is the container for storing the statistical data for predictions
        self.accuracy = []
        # this is the container for storing the accuracy for non-background predictions
        self.non_back_accuracy = []
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        print('Restoring weights from: ' + self.weights_file)
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.weights_file)


    def draw_result(self, img, result):
        for i in range(len(result)):
            # draw the cell with color code for object
            h_size = int(np.floor(img.shape[0] / self.cell_num))
            w_size = int(np.floor(img.shape[1] / self.cell_num))
            x = int(result[i][4]) * w_size
            y = int(result[i][5]) * h_size
            # determine the object type
            color_code = (0,0,0)
            if (result[i][0] == 0):
                # bud -> red
                color_code = (0, 0, 255)
                cv2.rectangle(img,(x,y),(x+w_size,y+h_size),color_code,3)
            elif (result[i][0] == 1):
                # flower -> green
                color_code = (0, 255, 0)
                cv2.rectangle(img,(x,y),(x+w_size,y+h_size),color_code,3)
            elif (result[i][0] == 2):
                # fruit -> blue
                color_code = (255, 0, 0)
                cv2.rectangle(img,(x,y),(x+w_size,y+h_size),color_code,3)

    def detect(self, img, line):
        img_h, img_w, _ = img.shape
        inputs = cv2.resize(img, (self.image_size, self.image_size))
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
        inputs = (inputs / 255.0) * 2.0 - 1.0
        inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3))

        result = self.detect_from_cvmat(inputs)[0]

        for i in range(len(result)):
            result[i][1] *= (1.0 * img_w / self.image_size)
            result[i][2] *= (1.0 * img_h / self.image_size)
        label = np.zeros((self.cell_num, self.cell_num, 2))
        label[:,:,1] = label[:,:,1]+3
 
        # construct the ground truth label
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
                    #print(x_cell,y_cell)
                    label[x_cell, y_cell, 0] = 1
                    #bud
                    label[x_cell, y_cell, 1] = 0
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
                    #print(x_cell,y_cell)
                    label[x_cell, y_cell, 0] = 1
                    # this is a flower
                    label[x_cell, y_cell, 1] = 1
        # draw flower
        for item in fruit:
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
                if label[x_cell, y_cell, 0] == 0:
                    #print(x_cell,y_cell)
                    label[x_cell, y_cell, 0] = 1
                    # this is a fruit
                    label[x_cell, y_cell, 1] = 2
        label = label[:,:,1]
 
        # loop through the result and compute the stats
        predicted_label = np.zeros((self.cell_num, self.cell_num))+3 #initialize everything as background point
        # correct non-background prediction
        correct = 0
        for i in range(len(result)):
            x_cell = int(result[i][4])
            y_cell = int(result[i][5])
            predicted_label[x_cell,y_cell] = result[i][0]
            # determine if this prediction is a background
            if(result[i][0] != 3):
                if(result[i][0] == label[x_cell,y_cell]):
                    correct += 1
        #compute the accuracy
        accuracy = np.sum((predicted_label == label).astype(int))/(self.cell_num * self.cell_num)
        self.accuracy.append(accuracy)
        nonback_accuracy = 0 if len(result)==0 else correct/len(result)
        self.non_back_accuracy.append(nonback_accuracy)
        return result


    def detect_from_cvmat(self, inputs):
        net_output = self.sess.run(self.net.logits,
                                   feed_dict={self.net.images: inputs})
        results = []
        for i in range(net_output.shape[0]):
            result = self.interpret_output(net_output[i])
            results.append(result)
        return results


    def interpret_output(self, output):
        # predicted conditional probabilities
        probs = np.zeros((self.cell_num, self.cell_num,
                          2, self.num_class))
        class_probs = np.reshape(
            output[0:self.boundary1],
            (self.cell_num, self.cell_num, self.num_class))
        #print(class_probs)
        # predicted object probability
        scales = np.reshape(
            output[self.boundary1:self.boundary2],
            (self.cell_num, self.cell_num, self.centers_per_cell))
        #print(scales)
        # predicted coordinate
        centers = np.reshape(
            output[self.boundary2:],
            (self.cell_num, self.cell_num, self.centers_per_cell, 4))
        #truncate the last width and height dimension
        centers = centers[:,:,:,0:2]
        offset = np.array(
            [np.arange(self.cell_num)] * self.cell_num * 1)
        offset = np.transpose(
            np.reshape(
                offset,
                [1, self.cell_num, self.cell_num]),
            (1, 2, 0))
        # scale the predictions back to the original input image size
        centers[:, :, :, 0] += offset
        centers[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        centers[:, :, :, :2] = 1.0 * centers[:, :, :, 0:2] / self.cell_num

        centers *= self.image_size

        for i in range(self.centers_per_cell):
            for j in range(self.num_class):
                probs[:, :, i, j] = np.multiply(
                    class_probs[:, :, j], scales[:, :, i])
        #compute the unconditional class probability and throw the predictions with low probility.
        #print(probs)
        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')

        filter_mat_centers = np.nonzero(filter_mat_probs)
        #print(filter_mat_centers[0],filter_mat_centers[1])
        centers_filtered = centers[filter_mat_centers[0],
                               filter_mat_centers[1], filter_mat_centers[2]]
        probs_filtered = probs[filter_mat_probs]
        #print(probs_filtered)
        classes_num_filtered = np.argmax(
            filter_mat_probs, axis=3)[filter_mat_centers[0], filter_mat_centers[1], filter_mat_centers[2]]
        #print(classes_num_filtered)
        # sort the probability from high to low
        result = []
        # format the output
        for i in range(len(centers_filtered)):
            #print(probs_filtered[i])
            result.append(
                [classes_num_filtered[i],      # integer class type
                 centers_filtered[i][0],
                 centers_filtered[i][1],
                 probs_filtered[i],
                 filter_mat_centers[0][i],
                 filter_mat_centers[1][i]])

        return result

    def calc_dist(self, center1, center2):
        distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
        cell_length = self.image_size/self.cell_num
        distance = distance / (2 * cell_length)
        distance = np.clip(distance, 0.0, 1.0)
        # clip the value
        return distance


    def image_detector(self, imname, line, wait=0):
        image = cv2.imread(imname)
        #get specie name
        specie = (imname.split("/")[2]).split(".")[0]
        result = self.detect(image, line)

        if (SAVE_IMG):
            self.draw_result(image, result)
            #save the output image
            #determine if the directory exists
            imname = imname.split("/")[2]
            cv2.imwrite("prediction/"+specie + '/' + imname, image)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='yolo.ckpt-30000', type=str)
    parser.add_argument('--weight_dir', default='output', type=str)
    parser.add_argument('--data_dir', default="network", type=str)
    parser.add_argument('--gpu', default='', type=str)
    args = parser.parse_args()
    
    categories = ['Anemone_canadensis','Anemone_hepatica','Aquilegia_canadensis','Bidens_vulgata','Celastrus_orbiculatus',
                  'Centaurea_stoebe','Cirsium_arvense','Cirsium_discolor','Geranium_maculatum','Geranium_robertianum',
                  'Hemerocallis_fulva','Hibiscus_moscheutos','Impatiens_capensis','Iris_pseudacorus']

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    for specie in categories:
        if not os.path.exists('prediction/'+specie):
            os.makedirs('prediction/'+specie)
        # construct the network and load trained weights
        tf.reset_default_graph() 
        
        yolo = YOLONet(False)
        weight_file = os.path.join('model', specie, args.weights)
        detector = Detector(yolo, weight_file)
        label_file = 'label/' + specie + '_label.txt'
        # read all the images
        print('Computing accuracy for ' + specie)
        with open(label_file, "r") as ground_truth:
            line = ground_truth.readline()
            while line:
                line = line.split(",")
                #get image name
                specie = line[0].split(".")[0]
                imname = "images/" +specie + "/" + line[0]
                # detect from image file
                detector.image_detector(imname, line)
                line = ground_truth.readline()
        accuracy = 0 if len(detector.accuracy) ==0 else np.sum(detector.accuracy)/len(detector.accuracy)
        print("Overall Accuracy: " + str(accuracy))
        non_back_accuracy = 0 if len(detector.accuracy) ==0 else np.sum(detector.non_back_accuracy)/len(detector.non_back_accuracy)
        print("None backgrond Accuracy: " + str(non_back_accuracy))
if __name__ == '__main__':
    main()
