# README #

The codes are for the paper "Convolutional Channel Features For Pedestrian, Face and Edge Detection".

Currently the codes include training and testing part for Caltech pedestrian detection. Models to produce curves in the paper are also provided.

The codes are written in Matlab codes, dependent on Caffe toolkit and Piotr's Computer Vision Matlab Toolbox. Codes are tested on Linux 12.04.3 LTS with 128GB memory and a Titan Z GPU card.

### Preparation ###

* Install Caffe with matCaffe (http://caffe.berkeleyvision.org)
* Download VGG-16 CaffeModel to './data/CaffeNets/' (https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md)
* Download Caltech Pedestrian Dataset (http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/) and set it up properly with codes in './data/code3.2.1'

### Use trained CCF model for pedestrian detection ###

* simply run './runDetect.m' and follow instructions in the codes, detection results will be save as 'allBBs.mat'

### Train a CCF pedestrian detector ###

* Run modified './toolbox-master/detector/acfDemoCal.m' to train a Acf detector and save the collected bbs for training
* Run './getFeat_train.m' to extract features of training data
* Run './trainModel.m' to train the boosting decision tree model
* Run './runDetect.m' using your model
* (Evaluation) With detection results available, modify './toolbox-master/detector/acfTest.m' yourself a little to evaluate the results, and './data/code3.2.1/dbEval.m' to plot curves in comparison state-of-the-arts

### Power law ###

* Power law can accelerate the detection time by a large margin with unnoticeable performance decrease (see paper for more details). Unfortunately, it doesn't hold for CCF on Caltech Pedestrian Dataset.
* You can check whether the power law holds for specific feature types on specific datasets using codes in './power_law'. Simply run 'getMean.m' and 'getLambda.m' in turn.

### Bug report ###

* If you find any bug or have any problems in running the codes, post it in Issues. I will try to help.

### Reference ###

* If you use our codes or models in your research, we are grateful if you cite the paper "Convolutional Channel Features For Pedestrian, Face and Edge Detection". 

### Acknowledgement ###

Much gratitude is presented to

* Piotr Dollar and his toolbox
* Caffe team
* VGG team