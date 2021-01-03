# YoloNet for plant feature detection
Plant speciemen feature detection using YOLOnet. Aims to detect features from 14 specimen categories
Code includes training, predicting for one image and evaluate the accuracy for all 14 categories on training sets.
Original work from https://github.com/hizhangp/yolo_tensorflow

## Installation Prerequisite
1. Linux system (I used Ubuntu 16.04, have not tested the code on any other Linux distributions)
2. Python 3.6, Tensorflow r1.10, OpenCV 3.4.2, CUDA(optional, you don't really need this for prediction and model evaluation) 

## Run the model

```
python predict.py (image_path) (specie name)
```
For example
```
python predict.py Anemone_canadensis_1111.jpg Anemone_canadensis
```
![alt text](https://github.com/BU-Spark/harvard-herbaria/blob/yolonet/example/Anemone_canadensis.1040272.17269.jpg)
The output file will be an image name as "output.jpg" with the feature bounded by colored square:
 - red: bud
 - green: flower
 - blue: fruit
 ### Evaluate the model
 You can also evaluate all the accuracies for all the models on the training sets. Run the following line:
```
python eval_model.py
```
which will compute the category-wise average overall accuracies and non-background prediction accuracies, visualize the detection result saving them to local folder named "predictions" , model by model.
