#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
 Authors:
 Radhika Kulkarni: Nearest Neighbor and Neural Network 
 Aniruddha Godbole: Adaboost and Best/Fourth model 

 This programs finds the correct orientation for an image file.

 (1) We have implemented four methods of classifying the image orientation.

   a) Nearest neighbor: This method calculates the euclidean distance between a test image vector and all the training image vectors 
    and assigns the orientation of closest training image to the test image.

   b) Adaboost: Three weak classifiers have been designed. Naive Bayesian 
    classifiers were used. One each for each color 
    i.e. Red, Blue and Green. While training: in addition to the hypothesis
    weights the weighted average image for each orientation is also computed.
    The weights are as determined by the Adaboost algorithm (in R&N).
    Such a average is intuitively a good representation because it 
    summarizes each orientation and given that the training set has all
    four orientations hence the only difference in the summarization for each 
    orientation is due to the orientation alone!
    By experimentation a noise level of 0.40 was found to give best results.
    Also, each pixel is considered to match with a given orientation if it's
    values falls within 4 sigma of the pixel value in the weighted average for
    that orientation. This again was found be experimentation.
    The error for my blue classifier, green classifier and red 
    classifier are 0.31, 0.53 and 0.63 respectively. 
    (Note: error is as defined in R&N.)
    As the errors for the green classifier and red classifier are greater than 
    0.5 the weights of these hypotheses are negative 
    (if the weights are calculated as per R&N).
   The hypothesis weights for my blue classifier, green classifier and red 
   classifier are 0.79, -0.15 and -0.53 respectively. These when linearly 
   adjusted by 0.53 gave a slighlty higher accuracy of 66.91% (On Python2 burrow; 
   it was 67.02% on my local machine with Python3. I guess one test image is 
   getting affected by the change in environment.
   Training with 36,976 images is executed in under 5 minutes. 
   Testing with 943 images is executed in a second.
   
   ADABOOST AND NAIVE BAYESIAN: It appears that Adaboost may not give an improvement
   when Naive Bayesian Classifiers are used as weak classifiers.
   Changes based  on using Adaboost with Naive Bayesian using techniques suggested in
   "A Study of AdaBoost with Naive Bayesian Classifiers: Weakness and Improvement"
   by Zijian Zheng, Microsoft; and other literature in the field may be consulted in
   the future.  
    About my other major Adaboost attempts:
    For my 1-vs-Rest attempt: I got an accuracy of 25.34%
    
    For six classifiers (each corresponding to a unique pair of orientations): 
    I got an accuracy of 32.13%.
    This seems to be happening because when I am training for more than one 
    orientation/label some kind of blurring seems to happen...which again seems
    to happen because of the way I have chosen to represent an average 
    orientation which I believe takes care of minor wrong labeling related 
    issues and which I found intuitive.
    
    For my attempt with three blue classifiers with different with different 
    discretization thresholds of 4 sigma, 2 sigma and 2 sigma (and with a noise 
    level 0.4) an accuracy of 67.16% was achieved. In this case the errors of 
    the three classifiers were all below 0.5. However, this ignored the green
    data and red data and the improvement did not seem statistically 
    significant and so perhaps this may not do better on other test data where
    possibly green and red are more important for the orientation. 
    Hence, this attempt was not used in this file.
    

   c) Neural  network: This method implements a fully connected  neural network with specified
    number of  hidden layers with backpropogation.
 
   d) Best/Fourth Model:I have also accidentally designed and then implemented
    an alternative (the Best/fourth model) with an accuracy of 66.80% but that
    trains in around 22 seconds. Here, I have used only the blue classifier
    which was the best weak classifier in the Adaboost implementation. So, this
    classifier is a Naive Bayesian classifier. 
    Again, by experimentation a noise level of 0.40 was found to give best results.
    Also, each pixel is considered to match with a given orientation if it's
    values falls within 4 sigma of the pixel value in the weighted average for
    that orientation. This again was found be experimentation. The training time
    for 36,976 images is around 22 seconds.
    Given that in comparison to my Adaboost implementation only one less test
    image is getting incorrectly classified and given that change in environment
    seems to have caused one less correct classification in the case of the 
    Adaboost implementation the Best/Fourth Model is being retained and not being
    substituted by the Adaboost implementation.

 (2) We have created two dictionaries of training and testing data.

 We have also created an output file with image id and its derived orientation.
 
 *We have experimented with all 3 methods and found that -- works best of all considering time and accuracy.
 
 *Average time of run for nearest  method is 5 mins for 943 images 
 *Average Training time for 36976 images is around 4 min 48 seconds for 36,976 images
 *Average time of run for adaboost method is around 1 second for 943 images. 
 *Average time of run for nnet method is 9 mins for 943 images
 *Average Training time for best/fourth model is around 22 seconds for 36,976 images
 *Average time of run for best/fourth model is around 1 seconds for 943 images

 *Accuracy of nearest method is around 67% 
 *Accuracy of adaboost method is 67.02% for the given test set 
 *Accuracy of nnet method is around 53 %


References:
Artificial Intelligence A Modern Approach, 2nd edition, Russell & Norvig, Page695 for the Adaboost implementation
https://www.pyimagesearch.com/2016/08/08/k-nn-classifier-for-image-classification/
http://shodhganga.inflibnet.ac.in/bitstream/10603/33597/12/12_chapter4.pdf
http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
http://www.bogotobogo.com/python/python_Neural_Networks_Backpropagation_for_XOR_using_one_hidden_layer.php

How to run program:
./orient.py train train_file.txt model_file.txt [model]
./orient.py test test_file.txt model_file.txt [model]

'''


from __future__ import division
from cStringIO import StringIO
import time
import sys
from sys import argv, maxint
import re
import numpy as np
import operator
import math
import os
import pickle #https://stackoverflow.com/questions/13906623/using-pickle-dump-typeerror-must-be-str-not-bytes

methods = [
    "nearest",
    "adaboost",
    "nnet",
    "best"
]


shape = (8, 8, 3)       # defined for  8×8×3 = 192 dimensional feature vector  as given in PDF
train_input = {}        # defined as dictionary
test_input = {}
id_lookup = {}
img_std = {}
str_write = ""


debug_train_limit = 36976         # This is set to 36976  and not 40000 as given in PDF for train data as its processing this much count 
debug_test_limit = 943         # This is set to 943 and not 1000 as given for test data count as its processing only thiscount
test_mode = False
model_file = ""
num_nnet_iter = 0



'''
Parsing of train data...
input_train_data() function takes entries from train data file and parses each entry for proper format .
So This function produces "train_input" dictionary with pixel information for each image file(along with its orientation) in train file.
'''

def input_train_data(train_file):

    with open(train_file, 'r') as f:          # Open train file and process each line
        num_example = 0                       # Initialise count to 0   
        num_parsed_example = 0

        for example in f:                    # For each line in file      
            num_example= num_example+1       # Increasement count for num_example      
            example = example.rstrip('\n')   # This returns one complete row  till next train statement 
                                             #eg:'train/10414509293.jpg 180 125 87 47 138 92 48 141 98 55 141 92 60 126 77 57 124 88 71 124 100 61 125 84 58 112 99 58 123 89 68 126 92 65 123 81 64 129 78 67 132 83 69 118 91 60 107 94 51 115 117 50 122 117 64 120 108 55 129 115 62 139 124 70 143 117 71 131 100 48 118 100 52 93 101 34 112 111 49 147 131 76 153 126 76 152 116 65 144 109 59 139 101 55 121 93 51 52 61 17 107 101 51 141 117 82 125 108 74 114 105 72 108 100 68 95 99 77 81 100 91 81 89 50 122 141 144 119 136 148 116 134 143 106 130 14^C0 111 135 146 115 140 160 104 130 151 150 180 200 152 186 218 147 182 214 143 179 211 138 174 206 137 173 205 145 177 205 132 169 202 133 177 217 126 171 210 122 167 208 118 163 206 113 159 204 108 156 201 103 151 198 102 150 198'
    
            entries = example.split(' ')     # This splits the line with spaces in word in the format like :'train/1040260450.jpg', '90', '193', '194', '190', '86', '85', '84', '137', '137', '137', '79', '76', '78', '56', '55', '58', '143', '139', '134', '100', '123', '160', '56', '96', '161', '210', '210', '207', '106', '105', '103', '140', '140', '140', '126', '125', '126', '69', '68', '72', '147', '144', '141', '121', '133', '156', '57', '97', '161', '218', '218', '214', '143', '143', '141', '127', '126', '125', '185', '185', '185', '82', '82', '84', '109', '107', '105', '100', '107', '122', '63', '101', '163', '219', '218', '213', '136', '136', '134', '118', '117', '115', '215', '216', '215', '86', '87', '91', '74', '73', '76', '114', '115', '122', '64', '97', '152', '218', '217', '212', '137', '136', '133', '144', '143', '141', '202', '203', '201', '169', '169', '170', '78', '77   

            example_name = entries.pop(0)           # It takes out first item from example list eg :train/1040260450.jpg
            example_id = re.search(r'[0-9]+', example_name).group() 

            id_lookup[example_name] = example_id       # This  ID is stored as only say "1040260450" from train/1040260450.jpg

            if example_name not in train_input:     # For all initial entries for dict of train input for each example name , blank entry is added  
                train_input[example_name] = {}


            orientation = int(entries.pop(0)) / 90        # Orientation is simplified by diving by 90 to convert to values of 0,1,2,3 for 0,90,180,270 respectively. 
            array = np.empty(shape)                       # New array created with 8*8*3 .shape[0]=8,shape[1]=8,shape[2]=3
            for idx, entry in enumerate(entries):
                i = int(idx / (shape[1] * shape[2]))       # idx count goes from 0 ,1,... and shape[1]=8 and shape[2]=3         
                j = int(idx % (shape[1] * shape[2]) / shape[2])
                k = int(idx % (shape[1] * shape[2]) % shape[2])
                array[i][j][k] = int(entry)
            train_input[example_name][orientation] = array            # train input stores this matrix of array for each image file and its orientation.
                                                                      

            if example_name not in img_std:
                img_std[example_name] = np.std(array)         #  Set img_std dict entries to array value
          
            num_parsed_example =num_parsed_example+ 1
            if num_example >= debug_train_limit:              # Break the loop if count exceeds count for train data
                break


     #   print ("Number of parsed examples:", num_parsed_example)
     #   print ("Processing Train File Complete!!") 
    return num_example, num_parsed_example





'''
Parsing of test input file.....

input_test_data() function parses the input file to create 8*8*3 matrix for each entry similar to train data and correct orientation given in the file 
This function produces "test_input" dictionary with pixel information for each image file in input test file.
'''

def input_test_data(test_file):
 
    with open(test_file, 'r') as f:                         # Open test file in read mode
        num_test = 0  
                                          
        for sample in f:                                     # For each line in test file,split the entries with spaces
            entries = sample.split(' ')        
            sample_name = entries.pop(0)                     # Get the sample name as first entry popuped from list eg:test/10008707066.jpg      
            sample_id = re.search(r'[0-9]+', sample_name).group()
            id_lookup[sample_name] = sample_id               # This  ID is stored as only say "1040260450" from train/1040260450.jpg 

            correct_orientation = int(entries.pop(0)) / 90    # Fetch (pop) the correct orientation as the next entry from entry list.
            array = np.empty(shape)                           # Create empty array of 8*8*3 size 
            for idx, entry in enumerate(entries):             # similar to train data ,it stores pixel information in matrix 
                i = int(idx / (shape[1] * shape[2]))                           
                j = int(idx % (shape[1] * shape[2]) / shape[2])
                k = int(idx % (shape[1] * shape[2]) % shape[2])
                array[i][j][k] = int(entry)
            test_input[sample_name] = (correct_orientation, array)       # Store this array information to test_input array for each sample file
            num_test =num_test+ 1                                        # Increment the number test   
            if num_test >= debug_test_limit:                             # If num_test exceeds 1000 (input file limit) then break 
                break
   # print("Number of samples processed",num_test)
   # print("Processing Input Test File Complete!!\n") 

   
 




# ---------------For Nearest neighbor- Calculating  Euclidean distance-------------------------------

test_files = {}
train_files = {}

orientation = [0, 90, 180, 270]

# This function reads the data from the file given as parameter and returns a numpy array of that data
def read_files(file_name):
    files = {}
    input_file = open(file_name, 'r')
    for line in input_file:
        data = line.split()
        img = np.empty(192, dtype=np.int)
        index = 2
        i = 0
        while i < 192:
            img[i] = int(data[index])
            index += 1
            i += 1
        files[data[0] + data[1]] = {"orient": int(data[1]), "img": img}

    input_file.close()
    return files

#This function trains the input file and create the model file for nearest neighbour
def train_nearest(train_file,model_file):   
    print("Traning Nearest..")       
    with open(train_file) as f:
      lines = f.readlines()
    with open(model_file, "w") as f1:
         f1.writelines(lines)  
    print("Model file is created with name",model_file)       
  
  
#This function takes the input test file and test is against the model file for nearest neighbour
def test_nearest(train_files,test_files):
    i = 0
    result = 0
    nearest_file_str = StringIO()
    for test_f_id in test_files:
        i += 1
        test_f_img = test_files[test_f_id]["img"]

        min_dist = maxint
        img_with_min_dist = ""

        for train_f_id in train_files:
            train_f_img = train_files[train_f_id]["img"]
            new_img = np.subtract(train_f_img, test_f_img)
            new_img = np.square(new_img)
            dist = np.sum(new_img)
            if dist < min_dist:
                min_dist = dist
                img_with_min_dist = train_f_id

        if test_files[test_f_id]["orient"] == train_files[img_with_min_dist]["orient"]:
            result += 1
        nearest_file_str.write(
            test_f_id.split('.jpg')[0] +".jpg"+" " + str(train_files[img_with_min_dist]["orient"]) + '\n')

    accuracy=str(result * 1.0*100 / (i * 1.0))
    return(nearest_file_str.getvalue(),accuracy)

#------------------------Nearest neighbor Ends------------------------------------------------------------



#---------------------- Neural Networks Section------------------------------------------------------------

if num_nnet_iter > 12000:
    target_values = [-0.995, 0.995]
else:
    target_values = [-0.99, 0.99]

def normalize_image(image):
    image = np.array(image)                             # Converted to format like ([[[  50.,  110.,  185.],[  54.,  122.,  199.],[ 110.,  134.,  163.],[  83.,   91.,   95.],[  44.,   46.,   25.], [ 133.,  112.,   68.],[ 165.,  139.,   84.], [ 159.,  133.,   76.]],
    return (image-np.mean(image))/np.std(image)         # Normalised array using formula: http://www.d.umn.edu/~deoka001/Normalization.html eg:[[-1.1641791 ,  0.06972864,  1.61211332], [-1.08191859,  0.31651019,  1.90002513],[ 0.06972864,  0.56329174,  1.15968048], [-0.48552984, -0.32100881, -0.2387483 ], [-1.28756988, 1.24643962,-1.67830733], [ 0.54272661,  0.1108589 , -0.79400678], [ 1.20081074,  0.66611738, -0.46496471], [ 1.07741996,  0.54272661, -0.62948575]],




def activation_func(input):
    return np.tanh(input)

# Derivative function for neural net
def derivative_func(back_input):
    if math.fabs(back_input) > 100:
        return 0
    temp = math.exp(back_input)
    temp = 2*temp/(1+temp**2)
    return temp**2

#Definding initial weights
def initialize_weights():
    weight1 = np.random.normal(loc=0.0, scale=1.0/math.sqrt(hidden_count), size=(192,hidden_count))
    weight2 = np.random.normal(loc=0.0, scale=0.5, size=(hidden_count,4))
    return weight1, weight2

#Implementing backpropogation with calculated weights ,image and orientation
def backpropagate(image, orientation, weight1, weight2, learn1, learn2):
    layer1 = [0] * hidden_count
    layer2 = [0] * 4
    output = [target_values[0]] * 4
    output[orientation] = [target_values[1]]
    a = {
        2 : [0] * 4,
        1 : [0] * hidden_count,
        0 : np.ndarray.flatten(np.array(image))
    }
    delta = {
        2 : [0] * 4,
        1 : [0] * hidden_count
    }
    for j in range(hidden_count):
        for i in range(192):
            layer1[j] += weight1[i][j] * a[0][i]
        a[1][j] = activation_func(layer1[j])
    for j in range(4):
        for i in range(hidden_count):
            layer2[j] += weight2[i][j] * a[1][i]
        a[2][j] = activation_func(layer2[j])
    debug = np.sum([math.fabs(output[j]-a[2][j]) for j in range(4)])
    debug2 = np.argmax([a[2][j] for j in range(4)])
    debug3 = np.zeros((192,hidden_count))
    debug4 = np.zeros((hidden_count, 4))
    if debug2 == orientation:
        debug2 = 1
    else:
        debug2 = 0
    for j in range(4):
        delta[2][j] = derivative_func(layer2[j]) * (output[j]-a[2][j])
    for i in range(hidden_count):
        for j in range(4):
            delta[1][i] += delta[2][j] * weight2[i][j]
        delta[1][i] *= derivative_func(layer1[i])
    for i in range(hidden_count):
        for j in range(4):
            weight2[i][j] += learn2 * a[1][i] * delta[2][j]
            debug4[i][j] += math.fabs(learn2 * a[1][i] * delta[2][j])
    for i in range(192):
        for j in range(hidden_count):
            weight1[i][j] += learn1 * a[0][i] * delta[1][j]
            debug3[i][j] += math.fabs(learn1 * a[0][i] * delta[1][j])

    return weight1, weight2, debug, debug2, debug3, debug4

#This functions trains the input file and creates model file with  array information on weights 
def nnet_train(model_file):
    example_index = []
    for example in train_input:
        example_index.append(example)
    num_example = len(example_index)
    w1, w2 = initialize_weights()
    print ("Training Neural Nets..")
    
    for t in range(num_nnet_iter):
        # implements stochastic gradient descent
        e = np.random.randint(0, num_example)
        o = np.random.randint(0, 4)
        image = normalize_image(train_input[example_index[e]][o])
        l1 = 0.5/(t+1)
        l2 = l1*0.7
        w1, w2, debug, debug2, debug3, debug4 = backpropagate(image, o, w1, w2, l1, l2)

    f = open(model_file, "w")
    pickle.dump(w1, f)
    pickle.dump(w2, f)
    f.close()
  
    print("Model file is created with name",model_file) 
   
#This function tests the input test file with model file created using train function 

def nnet_test(image, weight1, weight2):
    layer1 = [0] * hidden_count
    layer2 = [0] * 4
    a = {
        2: [0] * 4,
        1: [0] * hidden_count,
        0: np.ndarray.flatten(np.array(image))
    }
    for j in range(hidden_count):
        for i in range(192):
            layer1[j] += weight1[i][j] * a[0][i]
        a[1][j] = activation_func(layer1[j])
    for j in range(4):
        for i in range(hidden_count):
            layer2[j] += weight2[i][j] * a[1][i]
        a[2][j] = activation_func(layer2[j])
    return np.argmax([a[2][j] for j in range(4)])



#-----------------Neural Network section Ends ----------------------------------------------------------------------------------------------

#------------------Adaboost Section----------------------------------------------------------------------------------------------------------
def adaboost_blue_classifier(train_file,modelf):
    sigmul=4 #defined based on experimentation
    noise=0.4 #defined based on experimentation
    train_input = {}        # defined as dictionary
    id_lookup = {}
    shape = (8, 8, 3)       # defined for  8×8×3 = 192 dimensional feature vector  as given in PDF
    #Matrice containing the learning for the classifiers
    bluesumtrain0=np.zeros(shape)
    bluesumtrain1=np.zeros(shape)
    bluesumtrain2=np.zeros(shape)
    bluesumtrain3=np.zeros(shape)
    bluenumorient0=0
    bluenumorient1=0
    bluenumorient2=0
    bluenumorient3=0   
    ####Radhika's code for reading from file re-used
    #print("\nProcessing Train File..") 
    with open(train_file, 'r') as f:          # Open train file and process each line
        num_example = 0                       # Initialise count to 0   

        for example in f:                    # For each line in file      
            num_example= num_example+1       # Increasement count for num_example      
            example = example.rstrip('\n')   # This returns one complete row  till next train statement 
                                             #eg:'train/10414509293.jpg 180 125 87 47 138 92 48 141 98 55 141 92 60 126 77 57 124 88 71 124 100 61 125 84 58 112 99 58 123 89 68 126 92 65 123 81 64 129 78 67 132 83 69 118 91 60 107 94 51 115 117 50 122 117 64 120 108 55 129 115 62 139 124 70 143 117 71 131 100 48 118 100 52 93 101 34 112 111 49 147 131 76 153 126 76 152 116 65 144 109 59 139 101 55 121 93 51 52 61 17 107 101 51 141 117 82 125 108 74 114 105 72 108 100 68 95 99 77 81 100 91 81 89 50 122 141 144 119 136 148 116 134 143 106 130 14^C0 111 135 146 115 140 160 104 130 151 150 180 200 152 186 218 147 182 214 143 179 211 138 174 206 137 173 205 145 177 205 132 169 202 133 177 217 126 171 210 122 167 208 118 163 206 113 159 204 108 156 201 103 151 198 102 150 198'
    
            entries = example.split(' ')     # This splits the line with spaces in word in the format like :'train/1040260450.jpg', '90', '193', '194', '190', '86', '85', '84', '137', '137', '137', '79', '76', '78', '56', '55', '58', '143', '139', '134', '100', '123', '160', '56', '96', '161', '210', '210', '207', '106', '105', '103', '140', '140', '140', '126', '125', '126', '69', '68', '72', '147', '144', '141', '121', '133', '156', '57', '97', '161', '218', '218', '214', '143', '143', '141', '127', '126', '125', '185', '185', '185', '82', '82', '84', '109', '107', '105', '100', '107', '122', '63', '101', '163', '219', '218', '213', '136', '136', '134', '118', '117', '115', '215', '216', '215', '86', '87', '91', '74', '73', '76', '114', '115', '122', '64', '97', '152', '218', '217', '212', '137', '136', '133', '144', '143', '141', '202', '203', '201', '169', '169', '170', '78', '77   

            example_name = entries.pop(0)           # It takes out first item from example list eg :train/1040260450.jpg
            example_id = re.search(r'[0-9]+', example_name).group() 

            id_lookup[example_name] = example_id       # This  ID is stored as only say "1040260450" from train/1040260450.jpg

            if example_name not in train_input:     # For all initial entries for dict of train input for each example name , blank entry is added  
                train_input[example_name] = {}


            orientation = int(entries.pop(0)) / 90        # Orientation is simplified by diving by 90 to convert to values of 0,1,2,3 for 0,90,180,270 respectively. 
            array = np.empty(shape)                       # New array created with 8*8*3 .shape[0]=8,shape[1]=8,shape[2]=3
            for idx, entry in enumerate(entries):
                i = int(idx / (shape[1] * shape[2]))       # idx count goes from 0 ,1,... and shape[1]=8 and shape[2]=3         
                j = int(idx % (shape[1] * shape[2]) / shape[2])
                k = int(idx % (shape[1] * shape[2]) % shape[2])
                array[i][j][k] = int(entry)
            train_input[example_name][orientation] = array            # train input stores this matrix of array for each image file and its orientation.
            ####Radhika's code for reading from file re-used    
            if orientation==0:
                bluesumtrain0+=array
                bluenumorient0+=1
            elif orientation==1:
                bluesumtrain1+=array
                bluenumorient1+=1
            elif orientation==2:
                bluesumtrain2+=array
                bluenumorient2+=1
            elif orientation==3:
                bluesumtrain3+=array
                bluenumorient3+=1
            
        bluesumtrain0=bluesumtrain0/bluenumorient0 
        bluesumtrain1=bluesumtrain1/bluenumorient1 
        bluesumtrain2=bluesumtrain2/bluenumorient2 
        bluesumtrain3=bluesumtrain3/bluenumorient3 
        #print(bluesumtrain0) 
        #print(bluesumtrain0.std())

    bluelowsumtrain0=bluesumtrain0*(1-sigmul*bluesumtrain0.std()/bluesumtrain0.mean())
    bluehighsumtrain0=bluesumtrain0*(1+sigmul*bluesumtrain0.std()/bluesumtrain0.mean())
    bluelowsumtrain1=bluesumtrain1*(1-sigmul*bluesumtrain1.std()/bluesumtrain1.mean())
    bluehighsumtrain1=bluesumtrain1*(1+sigmul*bluesumtrain1.std()/bluesumtrain1.mean())
    bluelowsumtrain2=bluesumtrain2*(1-sigmul*bluesumtrain2.std()/bluesumtrain2.mean())
    bluehighsumtrain2=bluesumtrain2*(1+sigmul*bluesumtrain2.std()/bluesumtrain2.mean())
    bluelowsumtrain3=bluesumtrain3*(1-sigmul*bluesumtrain3.std()/bluesumtrain3.mean())
    bluehighsumtrain3=bluesumtrain3*(1+sigmul*bluesumtrain3.std()/bluesumtrain3.mean())
    #####################################################################
    #blue classifier   ---first loop give in R&N 
    with open(train_file, 'r') as f:          # Open train file and process each line
        example_weights={}#Aniruddha
        error=0 #Aniruddha
        for example in f:                    # For each line in file      
            example = example.rstrip('\n')   # This returns one complete row  till next train statement 
                                             #eg:'train/10414509293.jpg 180 125 87 47 138 92 48 141 98 55 141 92 60 126 77 57 124 88 71 124 100 61 125 84 58 112 99 58 123 89 68 126 92 65 123 81 64 129 78 67 132 83 69 118 91 60 107 94 51 115 117 50 122 117 64 120 108 55 129 115 62 139 124 70 143 117 71 131 100 48 118 100 52 93 101 34 112 111 49 147 131 76 153 126 76 152 116 65 144 109 59 139 101 55 121 93 51 52 61 17 107 101 51 141 117 82 125 108 74 114 105 72 108 100 68 95 99 77 81 100 91 81 89 50 122 141 144 119 136 148 116 134 143 106 130 14^C0 111 135 146 115 140 160 104 130 151 150 180 200 152 186 218 147 182 214 143 179 211 138 174 206 137 173 205 145 177 205 132 169 202 133 177 217 126 171 210 122 167 208 118 163 206 113 159 204 108 156 201 103 151 198 102 150 198'
    
            entries = example.split(' ')     # This splits the line with spaces in word in the format like :'train/1040260450.jpg', '90', '193', '194', '190', '86', '85', '84', '137', '137', '137', '79', '76', '78', '56', '55', '58', '143', '139', '134', '100', '123', '160', '56', '96', '161', '210', '210', '207', '106', '105', '103', '140', '140', '140', '126', '125', '126', '69', '68', '72', '147', '144', '141', '121', '133', '156', '57', '97', '161', '218', '218', '214', '143', '143', '141', '127', '126', '125', '185', '185', '185', '82', '82', '84', '109', '107', '105', '100', '107', '122', '63', '101', '163', '219', '218', '213', '136', '136', '134', '118', '117', '115', '215', '216', '215', '86', '87', '91', '74', '73', '76', '114', '115', '122', '64', '97', '152', '218', '217', '212', '137', '136', '133', '144', '143', '141', '202', '203', '201', '169', '169', '170', '78', '77   

            example_name = entries.pop(0)           # It takes out first item from example list eg :train/1040260450.jpg
            correct_orientation = int(entries.pop(0)) / 90    # Fetch (pop) the correct orientation as the next entry from entry list.
            example_weights[example_name,correct_orientation]=1/num_example #Aniruddha
            array=train_input[example_name][correct_orientation]
            p0,p1,p2,p3=0,0,0,0
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(2,3):
                        if array[i][j][k]<bluehighsumtrain0[i][j][k] and array[i][j][k]>bluelowsumtrain0[i][j][k]:
                            p0+=math.log(1-noise)
                        else:
                            p0+=math.log(noise)
                        if array[i][j][k]<bluehighsumtrain1[i][j][k] and array[i][j][k]>bluelowsumtrain1[i][j][k]:
                            p1+=math.log(1-noise)
                        else:
                            p1+=math.log(noise)
                        if array[i][j][k]<bluehighsumtrain2[i][j][k] and array[i][j][k]>bluelowsumtrain2[i][j][k]:
                            p2+=math.log(1-noise)
                        else:
                            p2+=math.log(noise)
                        if array[i][j][k]<bluehighsumtrain3[i][j][k] and array[i][j][k]>bluelowsumtrain3[i][j][k]:
                            p3+=math.log(1-noise)
                        else:
                            p3+=math.log(noise)
            maxvalue2=-999999999999999999
            maxindex2=-1
            p=[p0,p1,p2,p3]                
            for index in range(len(p)):
                if p[index]>maxvalue2:
                    maxvalue2=p[index]
                    maxindex2=index
            if correct_orientation!=maxindex2:
                error=error+example_weights[example_name,correct_orientation]
                        
            
    #Blue classifier   ---second loop 
    with open(train_file, 'r') as f:          # Open train file and process each line
        for example in f:                    # For each line in file      
            example = example.rstrip('\n')   # This returns one complete row  till next train statement 
                                             #eg:'train/10414509293.jpg 180 125 87 47 138 92 48 141 98 55 141 92 60 126 77 57 124 88 71 124 100 61 125 84 58 112 99 58 123 89 68 126 92 65 123 81 64 129 78 67 132 83 69 118 91 60 107 94 51 115 117 50 122 117 64 120 108 55 129 115 62 139 124 70 143 117 71 131 100 48 118 100 52 93 101 34 112 111 49 147 131 76 153 126 76 152 116 65 144 109 59 139 101 55 121 93 51 52 61 17 107 101 51 141 117 82 125 108 74 114 105 72 108 100 68 95 99 77 81 100 91 81 89 50 122 141 144 119 136 148 116 134 143 106 130 14^C0 111 135 146 115 140 160 104 130 151 150 180 200 152 186 218 147 182 214 143 179 211 138 174 206 137 173 205 145 177 205 132 169 202 133 177 217 126 171 210 122 167 208 118 163 206 113 159 204 108 156 201 103 151 198 102 150 198'
    
            entries = example.split(' ')     # This splits the line with spaces in word in the format like :'train/1040260450.jpg', '90', '193', '194', '190', '86', '85', '84', '137', '137', '137', '79', '76', '78', '56', '55', '58', '143', '139', '134', '100', '123', '160', '56', '96', '161', '210', '210', '207', '106', '105', '103', '140', '140', '140', '126', '125', '126', '69', '68', '72', '147', '144', '141', '121', '133', '156', '57', '97', '161', '218', '218', '214', '143', '143', '141', '127', '126', '125', '185', '185', '185', '82', '82', '84', '109', '107', '105', '100', '107', '122', '63', '101', '163', '219', '218', '213', '136', '136', '134', '118', '117', '115', '215', '216', '215', '86', '87', '91', '74', '73', '76', '114', '115', '122', '64', '97', '152', '218', '217', '212', '137', '136', '133', '144', '143', '141', '202', '203', '201', '169', '169', '170', '78', '77   

            example_name = entries.pop(0)           # It takes out first item from example list eg :train/1040260450.jpg
            correct_orientation = int(entries.pop(0)) / 90    # Fetch (pop) the correct orientation as the next entry from entry list.
            array=train_input[example_name][correct_orientation]
            p0,p1,p2,p3=0,0,0,0
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(2,3):
                        if array[i][j][k]<bluehighsumtrain0[i][j][k] and array[i][j][k]>bluelowsumtrain0[i][j][k]:
                            p0+=math.log(1-noise)
                        else:
                            p0+=math.log(noise)
                        if array[i][j][k]<bluehighsumtrain1[i][j][k] and array[i][j][k]>bluelowsumtrain1[i][j][k]:
                            p1+=math.log(1-noise)
                        else:
                            p1+=math.log(noise)
                        if array[i][j][k]<bluehighsumtrain2[i][j][k] and array[i][j][k]>bluelowsumtrain2[i][j][k]:
                            p2+=math.log(1-noise)
                        else:
                            p2+=math.log(noise)
                        if array[i][j][k]<bluehighsumtrain3[i][j][k] and array[i][j][k]>bluelowsumtrain3[i][j][k]:
                            p3+=math.log(1-noise)
                        else:
                            p3+=math.log(noise)
            maxvalue2=-999999999999999999
            maxindex2=-1
            p=[p0,p1,p2,p3]                
            for index in range(len(p)):
                if p[index]>maxvalue2:
                    maxvalue2=p[index]
                    maxindex2=index
            if correct_orientation!=maxindex2:
                pass
            else:
                example_weights[example_name,correct_orientation]=example_weights[example_name,correct_orientation]*error/(1-error)
        s=0
        for index in example_weights.keys():
            s+=example_weights[index]
    
        for index in example_weights.keys():
            example_weights[index]/=s
            #print(example_weights[index])
        #print('s=',s)
        print('blue error=',error)
        z2=math.log((1-error)/error)
    
    pickle.dump(bluehighsumtrain0,modelf)
    pickle.dump(bluelowsumtrain0,modelf)
    pickle.dump(bluehighsumtrain1,modelf)
    pickle.dump(bluelowsumtrain1,modelf)
    pickle.dump(bluehighsumtrain2,modelf)
    pickle.dump(bluelowsumtrain2,modelf)
    pickle.dump(bluehighsumtrain3,modelf)
    pickle.dump(bluelowsumtrain3,modelf)
    #pickle.dump(z2,modelf)
    #f.close()
    return z2,example_weights,modelf

def adaboost_green_classifier(train_file,example_weights,modelf):
    sigmul=4 #defined based on experimentation
    noise=0.4 #defined based on experimentation
    train_input = {}        # defined as dictionary
    id_lookup = {}
    shape = (8, 8, 3)       # defined for  8×8×3 = 192 dimensional feature vector  as given in PDF
    #Matrice containing the learning for the classifiers
    greensumtrain0=np.zeros(shape)
    greensumtrain1=np.zeros(shape)
    greensumtrain2=np.zeros(shape)
    greensumtrain3=np.zeros(shape)
    greennumorient0=0
    greennumorient1=0
    greennumorient2=0
    greennumorient3=0
    ####Radhika's code for reading from file re-used
    #print("\nProcessing Train File..") 
    with open(train_file, 'r') as f:          # Open train file and process each line
        num_example = 0                       # Initialise count to 0   

        for example in f:                    # For each line in file      
            num_example= num_example+1       # Increasement count for num_example      
            example = example.rstrip('\n')   # This returns one complete row  till next train statement 
                                             #eg:'train/10414509293.jpg 180 125 87 47 138 92 48 141 98 55 141 92 60 126 77 57 124 88 71 124 100 61 125 84 58 112 99 58 123 89 68 126 92 65 123 81 64 129 78 67 132 83 69 118 91 60 107 94 51 115 117 50 122 117 64 120 108 55 129 115 62 139 124 70 143 117 71 131 100 48 118 100 52 93 101 34 112 111 49 147 131 76 153 126 76 152 116 65 144 109 59 139 101 55 121 93 51 52 61 17 107 101 51 141 117 82 125 108 74 114 105 72 108 100 68 95 99 77 81 100 91 81 89 50 122 141 144 119 136 148 116 134 143 106 130 14^C0 111 135 146 115 140 160 104 130 151 150 180 200 152 186 218 147 182 214 143 179 211 138 174 206 137 173 205 145 177 205 132 169 202 133 177 217 126 171 210 122 167 208 118 163 206 113 159 204 108 156 201 103 151 198 102 150 198'
    
            entries = example.split(' ')     # This splits the line with spaces in word in the format like :'train/1040260450.jpg', '90', '193', '194', '190', '86', '85', '84', '137', '137', '137', '79', '76', '78', '56', '55', '58', '143', '139', '134', '100', '123', '160', '56', '96', '161', '210', '210', '207', '106', '105', '103', '140', '140', '140', '126', '125', '126', '69', '68', '72', '147', '144', '141', '121', '133', '156', '57', '97', '161', '218', '218', '214', '143', '143', '141', '127', '126', '125', '185', '185', '185', '82', '82', '84', '109', '107', '105', '100', '107', '122', '63', '101', '163', '219', '218', '213', '136', '136', '134', '118', '117', '115', '215', '216', '215', '86', '87', '91', '74', '73', '76', '114', '115', '122', '64', '97', '152', '218', '217', '212', '137', '136', '133', '144', '143', '141', '202', '203', '201', '169', '169', '170', '78', '77   

            example_name = entries.pop(0)           # It takes out first item from example list eg :train/1040260450.jpg
            example_id = re.search(r'[0-9]+', example_name).group() 

            id_lookup[example_name] = example_id       # This  ID is stored as only say "1040260450" from train/1040260450.jpg

            if example_name not in train_input:     # For all initial entries for dict of train input for each example name , blank entry is added  
                train_input[example_name] = {}


            orientation = int(entries.pop(0)) / 90        # Orientation is simplified by diving by 90 to convert to values of 0,1,2,3 for 0,90,180,270 respectively. 
            array = np.empty(shape)                       # New array created with 8*8*3 .shape[0]=8,shape[1]=8,shape[2]=3
            for idx, entry in enumerate(entries):
                i = int(idx / (shape[1] * shape[2]))       # idx count goes from 0 ,1,... and shape[1]=8 and shape[2]=3         
                j = int(idx % (shape[1] * shape[2]) / shape[2])
                k = int(idx % (shape[1] * shape[2]) % shape[2])
                array[i][j][k] = int(entry)
            train_input[example_name][orientation] = array            # train input stores this matrix of array for each image file and its orientation.
            ####Radhika's code for reading from file re-used
    with open(train_file, 'r') as f:          # Open train file and process each line
        num_example = 0                       # Initialise count to 0   

        for example in f:                    # For each line in file      
            num_example= num_example+1       # Increasement count for num_example      
            example = example.rstrip('\n')   # This returns one complete row  till next train statement 
                                             #eg:'train/10414509293.jpg 180 125 87 47 138 92 48 141 98 55 141 92 60 126 77 57 124 88 71 124 100 61 125 84 58 112 99 58 123 89 68 126 92 65 123 81 64 129 78 67 132 83 69 118 91 60 107 94 51 115 117 50 122 117 64 120 108 55 129 115 62 139 124 70 143 117 71 131 100 48 118 100 52 93 101 34 112 111 49 147 131 76 153 126 76 152 116 65 144 109 59 139 101 55 121 93 51 52 61 17 107 101 51 141 117 82 125 108 74 114 105 72 108 100 68 95 99 77 81 100 91 81 89 50 122 141 144 119 136 148 116 134 143 106 130 14^C0 111 135 146 115 140 160 104 130 151 150 180 200 152 186 218 147 182 214 143 179 211 138 174 206 137 173 205 145 177 205 132 169 202 133 177 217 126 171 210 122 167 208 118 163 206 113 159 204 108 156 201 103 151 198 102 150 198'
    
            entries = example.split(' ')     # This splits the line with spaces in word in the format like :'train/1040260450.jpg', '90', '193', '194', '190', '86', '85', '84', '137', '137', '137', '79', '76', '78', '56', '55', '58', '143', '139', '134', '100', '123', '160', '56', '96', '161', '210', '210', '207', '106', '105', '103', '140', '140', '140', '126', '125', '126', '69', '68', '72', '147', '144', '141', '121', '133', '156', '57', '97', '161', '218', '218', '214', '143', '143', '141', '127', '126', '125', '185', '185', '185', '82', '82', '84', '109', '107', '105', '100', '107', '122', '63', '101', '163', '219', '218', '213', '136', '136', '134', '118', '117', '115', '215', '216', '215', '86', '87', '91', '74', '73', '76', '114', '115', '122', '64', '97', '152', '218', '217', '212', '137', '136', '133', '144', '143', '141', '202', '203', '201', '169', '169', '170', '78', '77   

            example_name = entries.pop(0)           # It takes out first item from example list eg :train/1040260450.jpg
            example_id = re.search(r'[0-9]+', example_name).group() 

            id_lookup[example_name] = example_id       # This  ID is stored as only say "1040260450" from train/1040260450.jpg

            if example_name not in train_input:     # For all initial entries for dict of train input for each example name , blank entry is added  
                train_input[example_name] = {}


            orientation = int(entries.pop(0)) / 90        # Orientation is simplified by diving by 90 to convert to values of 0,1,2,3 for 0,90,180,270 respectively. 
            array = np.empty(shape)                       # New array created with 8*8*3 .shape[0]=8,shape[1]=8,shape[2]=3
            for idx, entry in enumerate(entries):
                i = int(idx / (shape[1] * shape[2]))       # idx count goes from 0 ,1,... and shape[1]=8 and shape[2]=3         
                j = int(idx % (shape[1] * shape[2]) / shape[2])
                k = int(idx % (shape[1] * shape[2]) % shape[2])
                array[i][j][k] = int(entry)*example_weights[example_name,orientation] ############WEIGHTING OF EXAMPLES############
            train_input[example_name][orientation] = array            # train input stores this matrix of array for each image file and its orientation.
            if orientation==0:
                greensumtrain0+=array
                greennumorient0+=1
            elif orientation==1:
                greensumtrain1+=array
                greennumorient1+=1
            elif orientation==2:
                greensumtrain2+=array
                greennumorient2+=1
            elif orientation==3:
                greensumtrain3+=array
                greennumorient3+=1
        greensumtrain0=greensumtrain0/greennumorient0 
        greensumtrain1=greensumtrain1/greennumorient1 
        greensumtrain2=greensumtrain2/greennumorient2 
        greensumtrain3=greensumtrain3/greennumorient3 

    greenlowsumtrain0=greensumtrain0*(1-sigmul*greensumtrain0.std()/greensumtrain0.mean())
    greenhighsumtrain0=greensumtrain0*(1+sigmul*greensumtrain0.std()/greensumtrain0.mean())
    greenlowsumtrain1=greensumtrain1*(1-sigmul*greensumtrain1.std()/greensumtrain1.mean())
    greenhighsumtrain1=greensumtrain1*(1+sigmul*greensumtrain1.std()/greensumtrain1.mean())
    greenlowsumtrain2=greensumtrain2*(1-sigmul*greensumtrain2.std()/greensumtrain2.mean())
    greenhighsumtrain2=greensumtrain2*(1+sigmul*greensumtrain2.std()/greensumtrain2.mean())
    greenlowsumtrain3=greensumtrain3*(1-sigmul*greensumtrain3.std()/greensumtrain3.mean())
    greenhighsumtrain3=greensumtrain3*(1+sigmul*greensumtrain3.std()/greensumtrain3.mean())

    #green classifier   ---first loop 
    with open(train_file, 'r') as f:          # Open train file and process each line
        error=0 #Aniruddha
        for example in f:                    # For each line in file      
            example = example.rstrip('\n')   # This returns one complete row  till next train statement 
                                             #eg:'train/10414509293.jpg 180 125 87 47 138 92 48 141 98 55 141 92 60 126 77 57 124 88 71 124 100 61 125 84 58 112 99 58 123 89 68 126 92 65 123 81 64 129 78 67 132 83 69 118 91 60 107 94 51 115 117 50 122 117 64 120 108 55 129 115 62 139 124 70 143 117 71 131 100 48 118 100 52 93 101 34 112 111 49 147 131 76 153 126 76 152 116 65 144 109 59 139 101 55 121 93 51 52 61 17 107 101 51 141 117 82 125 108 74 114 105 72 108 100 68 95 99 77 81 100 91 81 89 50 122 141 144 119 136 148 116 134 143 106 130 14^C0 111 135 146 115 140 160 104 130 151 150 180 200 152 186 218 147 182 214 143 179 211 138 174 206 137 173 205 145 177 205 132 169 202 133 177 217 126 171 210 122 167 208 118 163 206 113 159 204 108 156 201 103 151 198 102 150 198'
    
            entries = example.split(' ')     # This splits the line with spaces in word in the format like :'train/1040260450.jpg', '90', '193', '194', '190', '86', '85', '84', '137', '137', '137', '79', '76', '78', '56', '55', '58', '143', '139', '134', '100', '123', '160', '56', '96', '161', '210', '210', '207', '106', '105', '103', '140', '140', '140', '126', '125', '126', '69', '68', '72', '147', '144', '141', '121', '133', '156', '57', '97', '161', '218', '218', '214', '143', '143', '141', '127', '126', '125', '185', '185', '185', '82', '82', '84', '109', '107', '105', '100', '107', '122', '63', '101', '163', '219', '218', '213', '136', '136', '134', '118', '117', '115', '215', '216', '215', '86', '87', '91', '74', '73', '76', '114', '115', '122', '64', '97', '152', '218', '217', '212', '137', '136', '133', '144', '143', '141', '202', '203', '201', '169', '169', '170', '78', '77   

            example_name = entries.pop(0)           # It takes out first item from example list eg :train/1040260450.jpg
            correct_orientation = int(entries.pop(0)) / 90    # Fetch (pop) the correct orientation as the next entry from entry list.
            array=train_input[example_name][correct_orientation]
            p0,p1,p2,p3=0,0,0,0
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(1,2):
                        if array[i][j][k]<greenhighsumtrain0[i][j][k] and array[i][j][k]>greenlowsumtrain0[i][j][k]:
                            p0+=math.log(1-noise)
                        else:
                            p0+=math.log(noise)
                        if array[i][j][k]<greenhighsumtrain1[i][j][k] and array[i][j][k]>greenlowsumtrain1[i][j][k]:
                            p1+=math.log(1-noise)
                        else:
                            p1+=math.log(noise)
                        if array[i][j][k]<greenhighsumtrain2[i][j][k] and array[i][j][k]>greenlowsumtrain2[i][j][k]:
                            p2+=math.log(1-noise)
                        else:
                            p2+=math.log(noise)
                        if array[i][j][k]<greenhighsumtrain3[i][j][k] and array[i][j][k]>greenlowsumtrain3[i][j][k]:
                            p3+=math.log(1-noise)
                        else:
                            p3+=math.log(noise)
            maxvalue1=-999999999999999999
            maxindex1=-1
            p=[p0,p1,p2,p3]                
            for index in range(len(p)):
                if p[index]>maxvalue1:
                    maxvalue1=p[index]
                    maxindex1=index
            if correct_orientation!=maxindex1:
                error+=example_weights[example_name,correct_orientation]
                        
            
    #green classifier   ---second loop 
    with open(train_file, 'r') as f:          # Open train file and process each line
        for example in f:                    # For each line in file      
            example = example.rstrip('\n')   # This returns one complete row  till next train statement 
                                             #eg:'train/10414509293.jpg 180 125 87 47 138 92 48 141 98 55 141 92 60 126 77 57 124 88 71 124 100 61 125 84 58 112 99 58 123 89 68 126 92 65 123 81 64 129 78 67 132 83 69 118 91 60 107 94 51 115 117 50 122 117 64 120 108 55 129 115 62 139 124 70 143 117 71 131 100 48 118 100 52 93 101 34 112 111 49 147 131 76 153 126 76 152 116 65 144 109 59 139 101 55 121 93 51 52 61 17 107 101 51 141 117 82 125 108 74 114 105 72 108 100 68 95 99 77 81 100 91 81 89 50 122 141 144 119 136 148 116 134 143 106 130 14^C0 111 135 146 115 140 160 104 130 151 150 180 200 152 186 218 147 182 214 143 179 211 138 174 206 137 173 205 145 177 205 132 169 202 133 177 217 126 171 210 122 167 208 118 163 206 113 159 204 108 156 201 103 151 198 102 150 198'
    
            entries = example.split(' ')     # This splits the line with spaces in word in the format like :'train/1040260450.jpg', '90', '193', '194', '190', '86', '85', '84', '137', '137', '137', '79', '76', '78', '56', '55', '58', '143', '139', '134', '100', '123', '160', '56', '96', '161', '210', '210', '207', '106', '105', '103', '140', '140', '140', '126', '125', '126', '69', '68', '72', '147', '144', '141', '121', '133', '156', '57', '97', '161', '218', '218', '214', '143', '143', '141', '127', '126', '125', '185', '185', '185', '82', '82', '84', '109', '107', '105', '100', '107', '122', '63', '101', '163', '219', '218', '213', '136', '136', '134', '118', '117', '115', '215', '216', '215', '86', '87', '91', '74', '73', '76', '114', '115', '122', '64', '97', '152', '218', '217', '212', '137', '136', '133', '144', '143', '141', '202', '203', '201', '169', '169', '170', '78', '77   

            example_name = entries.pop(0)           # It takes out first item from example list eg :train/1040260450.jpg
            correct_orientation = int(entries.pop(0)) / 90    # Fetch (pop) the correct orientation as the next entry from entry list.
            array=train_input[example_name][correct_orientation]
            p0,p1,p2,p3=0,0,0,0
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(1,2):
                        if array[i][j][k]<greenhighsumtrain0[i][j][k] and array[i][j][k]>greenlowsumtrain0[i][j][k]:
                            p0+=math.log(1-noise)
                        else:
                            p0+=math.log(noise)
                        if array[i][j][k]<greenhighsumtrain1[i][j][k] and array[i][j][k]>greenlowsumtrain1[i][j][k]:
                            p1+=math.log(1-noise)
                        else:
                            p1+=math.log(noise)
                        if array[i][j][k]<greenhighsumtrain2[i][j][k] and array[i][j][k]>greenlowsumtrain2[i][j][k]:
                            p2+=math.log(1-noise)
                        else:
                            p2+=math.log(noise)
                        if array[i][j][k]<greenhighsumtrain3[i][j][k] and array[i][j][k]>greenlowsumtrain3[i][j][k]:
                            p3+=math.log(1-noise)
                        else:
                            p3+=math.log(noise)
            maxvalue1=-999999999999999999
            maxindex1=-1
            p=[p0,p1,p2,p3]                
            for index in range(len(p)):
                if p[index]>maxvalue1:
                    maxvalue1=p[index]
                    maxindex1=index
            if correct_orientation!=maxindex1:
                pass
            else:
                example_weights[example_name,correct_orientation]*=error/(1-error)
        s=0
        for index in example_weights.keys():
            s+=example_weights[index]
        for index in example_weights.keys():
            example_weights[index]=example_weights[index]/s
        print('green error=',error)
        z1=math.log((1-error)/error)
    
    pickle.dump(greenhighsumtrain0,modelf)
    pickle.dump(greenlowsumtrain0,modelf)
    pickle.dump(greenhighsumtrain1,modelf)
    pickle.dump(greenlowsumtrain1,modelf)
    pickle.dump(greenhighsumtrain2,modelf)
    pickle.dump(greenlowsumtrain2,modelf)
    pickle.dump(greenhighsumtrain3,modelf)
    pickle.dump(greenlowsumtrain3,modelf)
    #pickle.dump(z1,modelf)

    return z1,example_weights,modelf

    
def adaboost_red_classifier(train_file,example_weights,modelf):
    sigmul=4 #defined based on experimentation
    noise=0.4 #defined based on experimentation
    train_input = {}        # defined as dictionary
    id_lookup = {}
    shape = (8, 8, 3)       # defined for  8×8×3 = 192 dimensional feature vector  as given in PDF
    #Matrice containing the learning for the the classifiers

    redsumtrain0=np.zeros(shape)
    redsumtrain1=np.zeros(shape)
    redsumtrain2=np.zeros(shape)
    redsumtrain3=np.zeros(shape)
    rednumorient0=0
    rednumorient1=0
    rednumorient2=0
    rednumorient3=0  
    ####Radhika's code for reading from file re-used
    #print("\nProcessing Train File..") 
    with open(train_file, 'r') as f:          # Open train file and process each line
        num_example = 0                       # Initialise count to 0   

        for example in f:                    # For each line in file      
            num_example= num_example+1       # Increasement count for num_example      
            example = example.rstrip('\n')   # This returns one complete row  till next train statement 
                                             #eg:'train/10414509293.jpg 180 125 87 47 138 92 48 141 98 55 141 92 60 126 77 57 124 88 71 124 100 61 125 84 58 112 99 58 123 89 68 126 92 65 123 81 64 129 78 67 132 83 69 118 91 60 107 94 51 115 117 50 122 117 64 120 108 55 129 115 62 139 124 70 143 117 71 131 100 48 118 100 52 93 101 34 112 111 49 147 131 76 153 126 76 152 116 65 144 109 59 139 101 55 121 93 51 52 61 17 107 101 51 141 117 82 125 108 74 114 105 72 108 100 68 95 99 77 81 100 91 81 89 50 122 141 144 119 136 148 116 134 143 106 130 14^C0 111 135 146 115 140 160 104 130 151 150 180 200 152 186 218 147 182 214 143 179 211 138 174 206 137 173 205 145 177 205 132 169 202 133 177 217 126 171 210 122 167 208 118 163 206 113 159 204 108 156 201 103 151 198 102 150 198'
    
            entries = example.split(' ')     # This splits the line with spaces in word in the format like :'train/1040260450.jpg', '90', '193', '194', '190', '86', '85', '84', '137', '137', '137', '79', '76', '78', '56', '55', '58', '143', '139', '134', '100', '123', '160', '56', '96', '161', '210', '210', '207', '106', '105', '103', '140', '140', '140', '126', '125', '126', '69', '68', '72', '147', '144', '141', '121', '133', '156', '57', '97', '161', '218', '218', '214', '143', '143', '141', '127', '126', '125', '185', '185', '185', '82', '82', '84', '109', '107', '105', '100', '107', '122', '63', '101', '163', '219', '218', '213', '136', '136', '134', '118', '117', '115', '215', '216', '215', '86', '87', '91', '74', '73', '76', '114', '115', '122', '64', '97', '152', '218', '217', '212', '137', '136', '133', '144', '143', '141', '202', '203', '201', '169', '169', '170', '78', '77   

            example_name = entries.pop(0)           # It takes out first item from example list eg :train/1040260450.jpg
            example_id = re.search(r'[0-9]+', example_name).group() 

            id_lookup[example_name] = example_id       # This  ID is stored as only say "1040260450" from train/1040260450.jpg

            if example_name not in train_input:     # For all initial entries for dict of train input for each example name , blank entry is added  
                train_input[example_name] = {}


            orientation = int(entries.pop(0)) / 90        # Orientation is simplified by diving by 90 to convert to values of 0,1,2,3 for 0,90,180,270 respectively. 
            array = np.empty(shape)                       # New array created with 8*8*3 .shape[0]=8,shape[1]=8,shape[2]=3
            for idx, entry in enumerate(entries):
                i = int(idx / (shape[1] * shape[2]))       # idx count goes from 0 ,1,... and shape[1]=8 and shape[2]=3         
                j = int(idx % (shape[1] * shape[2]) / shape[2])
                k = int(idx % (shape[1] * shape[2]) % shape[2])
                array[i][j][k] = int(entry)
            train_input[example_name][orientation] = array            # train input stores this matrix of array for each image file and its orientation.
    with open(train_file, 'r') as f:          # Open train file and process each line
        num_example = 0                       # Initialise count to 0   

        for example in f:                    # For each line in file      
            num_example= num_example+1       # Increasement count for num_example      
            example = example.rstrip('\n')   # This returns one complete row  till next train statement 
                                             #eg:'train/10414509293.jpg 180 125 87 47 138 92 48 141 98 55 141 92 60 126 77 57 124 88 71 124 100 61 125 84 58 112 99 58 123 89 68 126 92 65 123 81 64 129 78 67 132 83 69 118 91 60 107 94 51 115 117 50 122 117 64 120 108 55 129 115 62 139 124 70 143 117 71 131 100 48 118 100 52 93 101 34 112 111 49 147 131 76 153 126 76 152 116 65 144 109 59 139 101 55 121 93 51 52 61 17 107 101 51 141 117 82 125 108 74 114 105 72 108 100 68 95 99 77 81 100 91 81 89 50 122 141 144 119 136 148 116 134 143 106 130 14^C0 111 135 146 115 140 160 104 130 151 150 180 200 152 186 218 147 182 214 143 179 211 138 174 206 137 173 205 145 177 205 132 169 202 133 177 217 126 171 210 122 167 208 118 163 206 113 159 204 108 156 201 103 151 198 102 150 198'
    
            entries = example.split(' ')     # This splits the line with spaces in word in the format like :'train/1040260450.jpg', '90', '193', '194', '190', '86', '85', '84', '137', '137', '137', '79', '76', '78', '56', '55', '58', '143', '139', '134', '100', '123', '160', '56', '96', '161', '210', '210', '207', '106', '105', '103', '140', '140', '140', '126', '125', '126', '69', '68', '72', '147', '144', '141', '121', '133', '156', '57', '97', '161', '218', '218', '214', '143', '143', '141', '127', '126', '125', '185', '185', '185', '82', '82', '84', '109', '107', '105', '100', '107', '122', '63', '101', '163', '219', '218', '213', '136', '136', '134', '118', '117', '115', '215', '216', '215', '86', '87', '91', '74', '73', '76', '114', '115', '122', '64', '97', '152', '218', '217', '212', '137', '136', '133', '144', '143', '141', '202', '203', '201', '169', '169', '170', '78', '77   

            example_name = entries.pop(0)           # It takes out first item from example list eg :train/1040260450.jpg
            example_id = re.search(r'[0-9]+', example_name).group() 

            id_lookup[example_name] = example_id       # This  ID is stored as only say "1040260450" from train/1040260450.jpg

            if example_name not in train_input:     # For all initial entries for dict of train input for each example name , blank entry is added  
                train_input[example_name] = {}


            orientation = int(entries.pop(0)) / 90        # Orientation is simplified by diving by 90 to convert to values of 0,1,2,3 for 0,90,180,270 respectively. 
            array = np.empty(shape)                       # New array created with 8*8*3 .shape[0]=8,shape[1]=8,shape[2]=3
            for idx, entry in enumerate(entries):
                i = int(idx / (shape[1] * shape[2]))       # idx count goes from 0 ,1,... and shape[1]=8 and shape[2]=3         
                j = int(idx % (shape[1] * shape[2]) / shape[2])
                k = int(idx % (shape[1] * shape[2]) % shape[2])
                array[i][j][k] = int(entry)*example_weights[example_name,orientation] ############WEIGHTING OF EXAMPLES############
            train_input[example_name][orientation] = array            # train input stores this matrix of array for each image file and its orientation.

            if orientation==0:
                redsumtrain0+=array
                rednumorient0+=1
            elif orientation==1:
                redsumtrain1+=array
                rednumorient1+=1
            elif orientation==2:
                redsumtrain2+=array
                rednumorient2+=1
            elif orientation==3:
                redsumtrain3+=array
                rednumorient3+=1
            
        redsumtrain0=redsumtrain0/rednumorient0 
        redsumtrain1=redsumtrain1/rednumorient1 
        redsumtrain2=redsumtrain2/rednumorient2 
        redsumtrain3=redsumtrain3/rednumorient3 

    redlowsumtrain0=redsumtrain0*(1-sigmul*redsumtrain0.std()/redsumtrain0.mean())
    redhighsumtrain0=redsumtrain0*(1+sigmul*redsumtrain0.std()/redsumtrain0.mean())
    redlowsumtrain1=redsumtrain1*(1-sigmul*redsumtrain1.std()/redsumtrain1.mean())
    redhighsumtrain1=redsumtrain1*(1+sigmul*redsumtrain1.std()/redsumtrain1.mean())
    redlowsumtrain2=redsumtrain2*(1-sigmul*redsumtrain2.std()/redsumtrain2.mean())
    redhighsumtrain2=redsumtrain2*(1+sigmul*redsumtrain2.std()/redsumtrain2.mean())
    redlowsumtrain3=redsumtrain3*(1-sigmul*redsumtrain3.std()/redsumtrain3.mean())
    redhighsumtrain3=redsumtrain3*(1+sigmul*redsumtrain3.std()/redsumtrain3.mean())
            
    #Red classifier   ---first loop 
    with open(train_file, 'r') as f:          # Open train file and process each line
        error=0 
        for example in f:                    # For each line in file      
            example = example.rstrip('\n')   # This returns one complete row  till next train statement 
                                             #eg:'train/10414509293.jpg 180 125 87 47 138 92 48 141 98 55 141 92 60 126 77 57 124 88 71 124 100 61 125 84 58 112 99 58 123 89 68 126 92 65 123 81 64 129 78 67 132 83 69 118 91 60 107 94 51 115 117 50 122 117 64 120 108 55 129 115 62 139 124 70 143 117 71 131 100 48 118 100 52 93 101 34 112 111 49 147 131 76 153 126 76 152 116 65 144 109 59 139 101 55 121 93 51 52 61 17 107 101 51 141 117 82 125 108 74 114 105 72 108 100 68 95 99 77 81 100 91 81 89 50 122 141 144 119 136 148 116 134 143 106 130 14^C0 111 135 146 115 140 160 104 130 151 150 180 200 152 186 218 147 182 214 143 179 211 138 174 206 137 173 205 145 177 205 132 169 202 133 177 217 126 171 210 122 167 208 118 163 206 113 159 204 108 156 201 103 151 198 102 150 198'
    
            entries = example.split(' ')     # This splits the line with spaces in word in the format like :'train/1040260450.jpg', '90', '193', '194', '190', '86', '85', '84', '137', '137', '137', '79', '76', '78', '56', '55', '58', '143', '139', '134', '100', '123', '160', '56', '96', '161', '210', '210', '207', '106', '105', '103', '140', '140', '140', '126', '125', '126', '69', '68', '72', '147', '144', '141', '121', '133', '156', '57', '97', '161', '218', '218', '214', '143', '143', '141', '127', '126', '125', '185', '185', '185', '82', '82', '84', '109', '107', '105', '100', '107', '122', '63', '101', '163', '219', '218', '213', '136', '136', '134', '118', '117', '115', '215', '216', '215', '86', '87', '91', '74', '73', '76', '114', '115', '122', '64', '97', '152', '218', '217', '212', '137', '136', '133', '144', '143', '141', '202', '203', '201', '169', '169', '170', '78', '77   

            example_name = entries.pop(0)           # It takes out first item from example list eg :train/1040260450.jpg
            
            correct_orientation = int(entries.pop(0)) / 90    # Fetch (pop) the correct orientation as the next entry from entry list.
            array=train_input[example_name][correct_orientation]
            p0,p1,p2,p3=0,0,0,0
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(0,1):
                        if array[i][j][k]<redhighsumtrain0[i][j][k] and array[i][j][k]>redlowsumtrain0[i][j][k]:
                            p0+=math.log(1-noise)
                        else:
                            p0+=math.log(noise)
                        if array[i][j][k]<redhighsumtrain1[i][j][k] and array[i][j][k]>redlowsumtrain1[i][j][k]:
                            p1+=math.log(1-noise)
                        else:
                            p1+=math.log(noise)
                        if array[i][j][k]<redhighsumtrain2[i][j][k] and array[i][j][k]>redlowsumtrain2[i][j][k]:
                            p2+=math.log(1-noise)
                        else:
                            p2+=math.log(noise)
                        if array[i][j][k]<redhighsumtrain3[i][j][k] and array[i][j][k]>redlowsumtrain3[i][j][k]:
                            p3+=math.log(1-noise)
                        else:
                            p3+=math.log(noise)
            maxvalue0=-999999999999999999
            maxindex0=-1
            p=[p0,p1,p2,p3]                
            for index in range(len(p)):
                if p[index]>maxvalue0:
                    maxvalue0=p[index]
                    maxindex0=index
            if correct_orientation!=maxindex0:
                error+=example_weights[example_name,correct_orientation]
                        
            
    #Red classifier   ---second loop 
    with open(train_file, 'r') as f:          # Open train file and process each line
        for example in f:                    # For each line in file      
            example = example.rstrip('\n')   # This returns one complete row  till next train statement 
                                             #eg:'train/10414509293.jpg 180 125 87 47 138 92 48 141 98 55 141 92 60 126 77 57 124 88 71 124 100 61 125 84 58 112 99 58 123 89 68 126 92 65 123 81 64 129 78 67 132 83 69 118 91 60 107 94 51 115 117 50 122 117 64 120 108 55 129 115 62 139 124 70 143 117 71 131 100 48 118 100 52 93 101 34 112 111 49 147 131 76 153 126 76 152 116 65 144 109 59 139 101 55 121 93 51 52 61 17 107 101 51 141 117 82 125 108 74 114 105 72 108 100 68 95 99 77 81 100 91 81 89 50 122 141 144 119 136 148 116 134 143 106 130 14^C0 111 135 146 115 140 160 104 130 151 150 180 200 152 186 218 147 182 214 143 179 211 138 174 206 137 173 205 145 177 205 132 169 202 133 177 217 126 171 210 122 167 208 118 163 206 113 159 204 108 156 201 103 151 198 102 150 198'
    
            entries = example.split(' ')     # This splits the line with spaces in word in the format like :'train/1040260450.jpg', '90', '193', '194', '190', '86', '85', '84', '137', '137', '137', '79', '76', '78', '56', '55', '58', '143', '139', '134', '100', '123', '160', '56', '96', '161', '210', '210', '207', '106', '105', '103', '140', '140', '140', '126', '125', '126', '69', '68', '72', '147', '144', '141', '121', '133', '156', '57', '97', '161', '218', '218', '214', '143', '143', '141', '127', '126', '125', '185', '185', '185', '82', '82', '84', '109', '107', '105', '100', '107', '122', '63', '101', '163', '219', '218', '213', '136', '136', '134', '118', '117', '115', '215', '216', '215', '86', '87', '91', '74', '73', '76', '114', '115', '122', '64', '97', '152', '218', '217', '212', '137', '136', '133', '144', '143', '141', '202', '203', '201', '169', '169', '170', '78', '77   

            example_name = entries.pop(0)           # It takes out first item from example list eg :train/1040260450.jpg
            correct_orientation = int(entries.pop(0)) / 90    # Fetch (pop) the correct orientation as the next entry from entry list.
            array=train_input[example_name][correct_orientation]
            p0,p1,p2,p3=0,0,0,0
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(0,1):
                        if array[i][j][k]<redhighsumtrain0[i][j][k] and array[i][j][k]>redlowsumtrain0[i][j][k]:
                            p0+=math.log(1-noise)
                        else:
                            p0+=math.log(noise)
                        if array[i][j][k]<redhighsumtrain1[i][j][k] and array[i][j][k]>redlowsumtrain1[i][j][k]:
                            p1+=math.log(1-noise)
                        else:
                            p1+=math.log(noise)
                        if array[i][j][k]<redhighsumtrain2[i][j][k] and array[i][j][k]>redlowsumtrain2[i][j][k]:
                            p2+=math.log(1-noise)
                        else:
                            p2+=math.log(noise)
                        if array[i][j][k]<redhighsumtrain3[i][j][k] and array[i][j][k]>redlowsumtrain3[i][j][k]:
                            p3+=math.log(1-noise)
                        else:
                            p3+=math.log(noise)
                maxvalue0=-999999999999999999
                maxindex0=-1
                p=[p0,p1,p2,p3]                
                for index in range(len(p)):
                    if p[index]>maxvalue0:
                        maxvalue0=p[index]
                        maxindex0=index
                if correct_orientation!=maxindex0:
                    pass
                else:
                    example_weights[example_name,correct_orientation]*=error/(1-error)
        s=0
        for index in example_weights.keys():
            s+=example_weights[index]
        for index in example_weights.keys():
            example_weights[index]=example_weights[index]/s
        print('red error=',error)
        z0=math.log((1-error)/(error))

    pickle.dump(redhighsumtrain0,modelf)
    pickle.dump(redlowsumtrain0,modelf)
    pickle.dump(redhighsumtrain1,modelf)
    pickle.dump(redlowsumtrain1,modelf)
    pickle.dump(redhighsumtrain2,modelf)
    pickle.dump(redlowsumtrain2,modelf)
    pickle.dump(redhighsumtrain3,modelf)
    pickle.dump(redlowsumtrain3,modelf)
    #pickle.dump(z0,modelf)
    pickle.dump(noise,modelf)

    return z0,example_weights,modelf
        
    
def adaboost_train_data(train_file,model_file):
    #modelf= open('Adaboost_model_file', 'wb')
    #modelf= open(model_file, 'wb') 
    modelf= open(model_file, 'w') #CHANGED FOR PYTHON2
    print('Adaboost Training has begun...')
    z2,example_weights,modelf=adaboost_blue_classifier(train_file,modelf)
    print('Training for first classifier i.e. BLue classifier is complete...')
    z1,example_weights,modelf=adaboost_green_classifier(train_file,example_weights,modelf)
    print('Training for second classifier i.e. Green classifier is complete...')
    z0,example_weights,modelf=adaboost_red_classifier(train_file,example_weights,modelf)
    print('Training for the last classifier i.e. Red classifier is complete...')
    print('Hypothesis weights for blue classifier, green classifier and red classifier are:',z2,z1,z0)   
    z=[z0,z1,z2]
    min=z[0]
    for each in z:
        if min>each:
            min=each
    if min<0:
        z0=z0-min
        z1=z1-min
        z2=z2-min
        print('Modified Hypothesis weights for blue classifier, green classifier and red classifier are:',z2,z1,z0)
    pickle.dump(z2,modelf)
    pickle.dump(z1,modelf)
    pickle.dump(z0,modelf)
    modelf.close()
    print('So, Adaboost training completed.')    
    return 1



def adaboost_test_data(test_file,model_file):    
    # read from file
    output=open('output.txt','w')
    #f = open('Adaboost_model_file', 'rb')
    #f = open(model_file, 'rb')
    f = open(model_file,'r') #CHANGED FOR PYTHON2
    bluehighsumtrain0=pickle.load(f)
    bluelowsumtrain0=pickle.load(f)  
    bluehighsumtrain1=pickle.load(f)
    bluelowsumtrain1=pickle.load(f)
    bluehighsumtrain2=pickle.load(f)
    bluelowsumtrain2=pickle.load(f)
    bluehighsumtrain3=pickle.load(f)
    bluelowsumtrain3=pickle.load(f)
    #z2=pickle.load(f)
    greenhighsumtrain0=pickle.load(f)
    greenlowsumtrain0=pickle.load(f)  
    greenhighsumtrain1=pickle.load(f)
    greenlowsumtrain1=pickle.load(f)
    greenhighsumtrain2=pickle.load(f)
    greenlowsumtrain2=pickle.load(f)
    greenhighsumtrain3=pickle.load(f)
    greenlowsumtrain3=pickle.load(f)
    #z1=pickle.load(f)
    redhighsumtrain0=pickle.load(f)
    redlowsumtrain0=pickle.load(f)  
    redhighsumtrain1=pickle.load(f)
    redlowsumtrain1=pickle.load(f)
    redhighsumtrain2=pickle.load(f)
    redlowsumtrain2=pickle.load(f)
    redhighsumtrain3=pickle.load(f)
    redlowsumtrain3=pickle.load(f)
    #z0=pickle.load(f)

    noise=pickle.load(f)
    z2=pickle.load(f)
    z1=pickle.load(f)
    z0=pickle.load(f)
    f.close()
    #z2=1.3261112339300083
    #z1=0.38229057554062906
    #z0=0.0
    test_input = {}
    id_lookup={}
    shape = (8, 8, 3)       # defined for  8×8×3 = 192 dimensional feature vector  as given in PDF
    #print("\nProcessing Input Test File..")
    with open(test_file, 'r') as f:                         # Open test file in read mode
        num_test = 0                
        correctcount=0 #Aniruddha
        for sample in f:                                     # For each line in test file,split the entries with spaces
            entries = sample.split(' ')        
            sample_name = entries.pop(0)                     # Get the sample name as first entry popuped from list eg:test/10008707066.jpg      
            sample_id = re.search(r'[0-9]+', sample_name).group()
            id_lookup[sample_name] = sample_id               # This  ID is stored as only say "1040260450" from train/1040260450.jpg 

            correct_orientation = int(entries.pop(0)) / 90    # Fetch (pop) the correct orientation as the next entry from entry list.
            array = np.empty(shape)                           # Create empty array of 8*8*3 size 
            for idx, entry in enumerate(entries):             # similar to train data ,it stores pixel information in matrix 
                i = int(idx / (shape[1] * shape[2]))                           
                j = int(idx % (shape[1] * shape[2]) / shape[2])
                k = int(idx % (shape[1] * shape[2]) % shape[2])
                array[i][j][k] = int(entry)
            test_input[sample_name] = (correct_orientation, array)       # Store this array information to test_input array for each sample file
            num_test =num_test+ 1                                        # Increment the number test   

            #classifier2
            p0,p1,p2,p3=0,0,0,0
            for i in range(shape[0]):
                for j in range(shape[1]):
                        k=2
                        if array[i][j][k]<bluehighsumtrain0[i][j][k] and array[i][j][k]>bluelowsumtrain0[i][j][k]:
                            p0+=math.log(1-noise)*z2
                        else:
                            p0+=math.log(noise)*z2
                        if array[i][j][k]<bluehighsumtrain1[i][j][k] and array[i][j][k]>bluelowsumtrain1[i][j][k]:
                            p1+=math.log(1-noise)*z2
                        else:
                            p1+=math.log(noise)*z2
                        if array[i][j][k]<bluehighsumtrain2[i][j][k] and array[i][j][k]>bluelowsumtrain2[i][j][k]:
                            p2+=math.log(1-noise)*z2
                        else:
                            p2+=math.log(noise)*z2
                        if array[i][j][k]<bluehighsumtrain3[i][j][k] and array[i][j][k]>bluelowsumtrain3[i][j][k]:
                            p3+=math.log(1-noise)*z2
                        else:
                            p3+=math.log(noise)*z2

            for i in range(shape[0]):
                for j in range(shape[1]):
                        k=1
                        if array[i][j][k]<greenhighsumtrain0[i][j][k] and array[i][j][k]>greenlowsumtrain0[i][j][k]:
                            p0+=math.log(1-noise)*z1
                        else:
                            p0+=math.log(noise)*z1
                        if array[i][j][k]<greenhighsumtrain1[i][j][k] and array[i][j][k]>greenlowsumtrain1[i][j][k]:
                            p1+=math.log(1-noise)*z1
                        else:
                            p1+=math.log(noise)*z1
                        if array[i][j][k]<greenhighsumtrain2[i][j][k] and array[i][j][k]>greenlowsumtrain2[i][j][k]:
                            p2+=math.log(1-noise)*z1
                        else:
                            p2+=math.log(noise)*z1
                        if array[i][j][k]<greenhighsumtrain3[i][j][k] and array[i][j][k]>greenlowsumtrain3[i][j][k]:
                            p3+=math.log(1-noise)*z1
                        else:
                            p3+=math.log(noise)*z1
            for i in range(shape[0]):
                for j in range(shape[1]):
                        k=0
                        if array[i][j][k]<redhighsumtrain0[i][j][k] and array[i][j][k]>redlowsumtrain0[i][j][k]:
                            p0+=math.log(1-noise)*z0
                        else:
                            p0+=math.log(noise)*z0
                        if array[i][j][k]<redhighsumtrain1[i][j][k] and array[i][j][k]>redlowsumtrain1[i][j][k]:
                            p1+=math.log(1-noise)*z0
                        else:
                            p1+=math.log(noise)*z0
                        if array[i][j][k]<redhighsumtrain2[i][j][k] and array[i][j][k]>redlowsumtrain2[i][j][k]:
                            p2+=math.log(1-noise)*z0
                        else:
                            p2+=math.log(noise)*z0
                        if array[i][j][k]<redhighsumtrain3[i][j][k] and array[i][j][k]>redlowsumtrain3[i][j][k]:
                            p3+=math.log(1-noise)*z0
                        else:
                            p3+=math.log(noise)*z0
            maxvalue=-999999999999999999
            maxindex=0
            p=[p0,p1,p2,p3]                
            for index in range(len(p)):
                if p[index]>maxvalue:
                    maxvalue=p[index]
                    maxindex=index
            #print(correct_orientation,maxindex)
            ####output file###########
            #output.write(sample_name+' '+str(int(correct_orientation*90))+' '+str(int(maxindex*90))+'\n')
            output.write(sample_name+' '+str(int(maxindex*90))+'\n')
            if correct_orientation==maxindex:
                correctcount+=1
    print('Processing Input Test File Complete')                
    print('Number of test images processed',num_test)
    print('Correct count=',correctcount)
    print('Accuracy=',100*correctcount/num_test,'%')
    output.close()        
    return 1
    
def train_adaboost(train_file,model_file):
    #train_file = "train-data.txt" # Training file name
    adaboost_train_data(train_file,model_file) 
    return 1

def test_adaboost(test_file,model_file):
    #test_file =  "test-data.txt"  # Test file name  
    adaboost_test_data(test_file,model_file)
    return 1



#------------------Adaboost Section Ends----------------------------------------------------------------------------------------------------------
#------------------Best/Fourth Model Section Begins----------------------------------------------------------------------------------------------------------
#def fourth_train_data(train_file):
def fourth_train_data(train_file,model_file):
    #model4f= open('fourthmodel_file', 'wb')
    #model4f= open(model_file, 'wb')
    model4f = open(model_file, 'w') #CHANGED FOR PYTHON2
    sigmul=4 #defined based on experimentation
    noise=0.4 #defined based on experimentation
    train_input = {}        # defined as dictionary
    id_lookup = {}
    shape = (8, 8, 3)       # defined for  8×8×3 = 192 dimensional feature vector  as given in PDF
    #Matrice containing the learning for the classifiers

    sumtrain0=np.zeros(shape)
    sumtrain1=np.zeros(shape)
    sumtrain2=np.zeros(shape)
    sumtrain3=np.zeros(shape)
    numorient0=0
    numorient1=0
    numorient2=0
    numorient3=0
    ####Radhika's code for reading from file re-used
    print("\nProcessing Training File..") 
    with open(train_file, 'r') as f:          # Open train file and process each line
        num_example = 0                       # Initialise count to 0   


        for example in f:                    # For each line in file      
            num_example= num_example+1       # Increasement count for num_example      
            example = example.rstrip('\n')   # This returns one complete row  till next train statement 
                                             #eg:'train/10414509293.jpg 180 125 87 47 138 92 48 141 98 55 141 92 60 126 77 57 124 88 71 124 100 61 125 84 58 112 99 58 123 89 68 126 92 65 123 81 64 129 78 67 132 83 69 118 91 60 107 94 51 115 117 50 122 117 64 120 108 55 129 115 62 139 124 70 143 117 71 131 100 48 118 100 52 93 101 34 112 111 49 147 131 76 153 126 76 152 116 65 144 109 59 139 101 55 121 93 51 52 61 17 107 101 51 141 117 82 125 108 74 114 105 72 108 100 68 95 99 77 81 100 91 81 89 50 122 141 144 119 136 148 116 134 143 106 130 14^C0 111 135 146 115 140 160 104 130 151 150 180 200 152 186 218 147 182 214 143 179 211 138 174 206 137 173 205 145 177 205 132 169 202 133 177 217 126 171 210 122 167 208 118 163 206 113 159 204 108 156 201 103 151 198 102 150 198'
    
            entries = example.split(' ')     # This splits the line with spaces in word in the format like :'train/1040260450.jpg', '90', '193', '194', '190', '86', '85', '84', '137', '137', '137', '79', '76', '78', '56', '55', '58', '143', '139', '134', '100', '123', '160', '56', '96', '161', '210', '210', '207', '106', '105', '103', '140', '140', '140', '126', '125', '126', '69', '68', '72', '147', '144', '141', '121', '133', '156', '57', '97', '161', '218', '218', '214', '143', '143', '141', '127', '126', '125', '185', '185', '185', '82', '82', '84', '109', '107', '105', '100', '107', '122', '63', '101', '163', '219', '218', '213', '136', '136', '134', '118', '117', '115', '215', '216', '215', '86', '87', '91', '74', '73', '76', '114', '115', '122', '64', '97', '152', '218', '217', '212', '137', '136', '133', '144', '143', '141', '202', '203', '201', '169', '169', '170', '78', '77   

            example_name = entries.pop(0)           # It takes out first item from example list eg :train/1040260450.jpg
            example_id = re.search(r'[0-9]+', example_name).group() 

            id_lookup[example_name] = example_id       # This  ID is stored as only say "1040260450" from train/1040260450.jpg

            if example_name not in train_input:     # For all initial entries for dict of train input for each example name , blank entry is added  
                train_input[example_name] = {}


            orientation = int(entries.pop(0)) / 90        # Orientation is simplified by diving by 90 to convert to values of 0,1,2,3 for 0,90,180,270 respectively. 
            array = np.empty(shape)                       # New array created with 8*8*3 .shape[0]=8,shape[1]=8,shape[2]=3
            for idx, entry in enumerate(entries):
                i = int(idx / (shape[1] * shape[2]))       # idx count goes from 0 ,1,... and shape[1]=8 and shape[2]=3         
                j = int(idx % (shape[1] * shape[2]) / shape[2])
                k = int(idx % (shape[1] * shape[2]) % shape[2])
                array[i][j][k] = int(entry)
            train_input[example_name][orientation] = array            # train input stores this matrix of array for each image file and its orientation.

            if orientation==0:
                sumtrain0+=array
                numorient0+=1
            elif orientation==1:
                sumtrain1+=array
                numorient1+=1
            elif orientation==2:
                sumtrain2+=array
                numorient2+=1
            elif orientation==3:
                sumtrain3+=array
                numorient3+=1
            
        sumtrain0=sumtrain0/numorient0 
        sumtrain1=sumtrain1/numorient1 
        sumtrain2=sumtrain2/numorient2 
        sumtrain3=sumtrain3/numorient3 
        print ('Number of training images:', num_example)
        # print(train_input)
        #print(sumtrain0) #Aniruddha
        #print(sumtrain0.std())#Aniruddha
    
    pickle.dump(sumtrain0,model4f)
    pickle.dump(sumtrain1,model4f)
    pickle.dump(sumtrain2,model4f)
    pickle.dump(sumtrain3,model4f)
    pickle.dump(shape,model4f) 
    pickle.dump(sigmul,model4f) 
    pickle.dump(noise,model4f)        
    model4f.close()
    print('Fourth Model training completed.')
    return 1



def fourth_test_data(test_file,model_file):
    print("\nProcessing Input Test File..")
    test_input={}
    id_lookup={}
    output=open('output.txt','w')
    #model4f = open('fourthmodel_file', 'rb')
    #model4f = open(model_file, 'rb')
    model4f = open(model_file, 'r') #CHANGED FOR PYTHON2
    
    sumtrain0=pickle.load(model4f)
    sumtrain1=pickle.load(model4f)
    sumtrain2=pickle.load(model4f)
    sumtrain3=pickle.load(model4f)
    shape=pickle.load(model4f)
    sigmul=pickle.load(model4f)
    noise=pickle.load(model4f)
    model4f.close()
    ####Radhika's code for reading from file re-used
    with open(test_file, 'r') as f:                         # Open test file in read mode
        num_test = 0                
        correctcount=0 
        for sample in f:                                     # For each line in test file,split the entries with spaces
            entries = sample.split(' ')        
            sample_name = entries.pop(0)                     # Get the sample name as first entry popuped from list eg:test/10008707066.jpg      
            sample_id = re.search(r'[0-9]+', sample_name).group()
            id_lookup[sample_name] = sample_id               # This  ID is stored as only say "1040260450" from train/1040260450.jpg 

            correct_orientation = int(entries.pop(0)) / 90    # Fetch (pop) the correct orientation as the next entry from entry list.
            array = np.empty(shape)                           # Create empty array of 8*8*3 size 
            for idx, entry in enumerate(entries):             # similar to train data ,it stores pixel information in matrix 
                i = int(idx / (shape[1] * shape[2]))                           
                j = int(idx % (shape[1] * shape[2]) / shape[2])
                k = int(idx % (shape[1] * shape[2]) % shape[2])
                array[i][j][k] = int(entry)
            test_input[sample_name] = (correct_orientation, array)       # Store this array information to test_input array for each sample file
            num_test =num_test+ 1                                        # Increment the number test   
            p0,p1,p2,p3=0,0,0,0
            lowsumtrain0=sumtrain0*(1-sigmul*sumtrain0.std()/sumtrain0.mean())
            highsumtrain0=sumtrain0*(1+sigmul*sumtrain0.std()/sumtrain0.mean())
            lowsumtrain1=sumtrain1*(1-sigmul*sumtrain1.std()/sumtrain1.mean())
            highsumtrain1=sumtrain1*(1+sigmul*sumtrain1.std()/sumtrain1.mean())
            lowsumtrain2=sumtrain2*(1-sigmul*sumtrain2.std()/sumtrain2.mean())
            highsumtrain2=sumtrain2*(1+sigmul*sumtrain2.std()/sumtrain2.mean())
            lowsumtrain3=sumtrain3*(1-sigmul*sumtrain3.std()/sumtrain3.mean())
            highsumtrain3=sumtrain3*(1+sigmul*sumtrain3.std()/sumtrain3.mean())
            for i in range(shape[0]):
                for j in range(shape[1]):
                    k=2
                    if array[i][j][k]<highsumtrain0[i][j][k] and array[i][j][k]>lowsumtrain0[i][j][k]:
                        p0+=math.log(1-noise)
                    else:
                        p0+=math.log(noise)
                    if array[i][j][k]<highsumtrain1[i][j][k] and array[i][j][k]>lowsumtrain1[i][j][k]:
                        p1+=math.log(1-noise)
                    else:
                        p1+=math.log(noise)
                    if array[i][j][k]<highsumtrain2[i][j][k] and array[i][j][k]>lowsumtrain2[i][j][k]:
                        p2+=math.log(1-noise)
                    else:
                        p2+=math.log(noise)
                    if array[i][j][k]<highsumtrain3[i][j][k] and array[i][j][k]>lowsumtrain3[i][j][k]:
                        p3+=math.log(1-noise)
                    else:
                        p3+=math.log(noise)
            maxvalue=-999999999999999999
            maxindex=0
            p=[p0,p1,p2,p3]                
            for index in range(len(p)):
                if p[index]>maxvalue:
                    maxvalue=p[index]
                    maxindex=index
            #print(correct_orientation,maxindex)
            ####output file###########
            #output.write(sample_name+' '+str(int(correct_orientation*90))+' '+str(int(maxindex*90))+'\n')
            output.write(sample_name+' '+str(int(maxindex*90))+'\n')
            if correct_orientation==maxindex:
                correctcount+=1
    print('Processing Input Test File Complete')                
    print('Number of test images processed',num_test)
    print('Correct count=',correctcount)
    print('Accuracy=',100*correctcount/num_test,'%')
    output.close()        
    return 1

def train_fourth(train_file,model_file):
    print('Fourth model training')
    #train_file = "train-data.txt" # Training file name
    #fourth_train_data(train_file)
    fourth_train_data(train_file,model_file)
    return 1

def test_fourth(test_file,model_file):
    print('Fourth model testing')
    #test_file =  "test-data.txt"  # Test file name  
    #fourth_test_data(test_file)
    fourth_test_data(test_file,model_file)
    return 1


#------------------Best/Fourth Model Section Ends----------------------------------------------------------------------------------------------------------

#--------------------Main Section----------------------------------------------------------------------------------------------------------

if __name__ == "__main__":  
 hidden_count=10
 if(sys.argv[1] =="train"):                      # If train command 
     
     train_file = sys.argv[2]                     # Get the train file name         
     model_file = sys.argv[3]                      # Get the model file name   
     method = sys.argv[4]                         # Get the method name from list of nearest,adaboost,nnet,best    
     
     if(method=="nearest"):
           train_nearest(train_file,model_file)
    
     elif(method=="nnet"):
         num_example, num_parsed_example = input_train_data(train_file)   # This we have to call for storing data in dict format for further processing.
         num_nnet_iter = int(4e5/hidden_count) 
         nnet_train(model_file)   
     
     elif(method=="adaboost"):
         train_adaboost(train_file,model_file) 

     elif(method=="best"):
         train_fourth(train_file,model_file) 
                

     
 elif(sys.argv[1] =="test"):                       # If test command        
     test_file = sys.argv[2]                       # Get the test file name         
     model_file = sys.argv[3]                      # Get the model file name   
     method = sys.argv[4]                          # Get the method name from list of nearest,adaboost,nnet,best    
     
     if(method=="nearest"):
            train_files = read_files(model_file)
            test_files = read_files(test_file)
            str_write,accuracy=test_nearest(train_files,test_files) 
            print("Accuracy=",accuracy)                 # Print Accuracy of each method
            with open('output.txt', 'w') as f:
              f.write(str_write)
            print("Output is written to output.txt")                    # Print the output to file
    
     elif(method=="nnet"):

           num_correct =0 
           input_test_data(test_file)   # This we have to call for storing data in dict format for further processing.
                                        # Run the test function for nnet
           f = open(model_file)
           w1=pickle.load(f)
           w2=pickle.load(f)
           f.close()
   
           for sample in test_input:
             correct_orientation = test_input[sample][0]                         # Get the correct orientation    
             image = normalize_image(test_input[sample][1])                      # Get the image array in normalized format 
             detected_orientation = nnet_test(image, w1, w2)                      # Get the detected orientation by running nnet run 
             str_write += sample + " " + str(detected_orientation*90) + "\n"     # Store the values in output file 
             if detected_orientation == correct_orientation:                     # If detected orientation matches correct one ,increment the count  
               num_correct += 1
    
    
           accuracy=(num_correct/943)*100
           print("Accuracy=",accuracy)                 # Print Accuracy
           with open('output.txt', 'w') as f:
              f.write(str_write)
           print("Output is written to output.txt")  

     elif(method=="adaboost"):     # If we decide to use adaboost as best method
           test_adaboost(test_file,model_file) 

     elif(method=="best"):     # If we decide to use adaboost as best method
           test_fourth(test_file,model_file)     
