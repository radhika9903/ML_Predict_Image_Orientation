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
