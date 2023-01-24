import cv2
import numpy as np
import os
import glob
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from featureextractionhelper import ExtractionHelper

#Config:
isNormalizeFeaturesTrue = False
isShowPredictionsEnabled = False

# select which feature to use
extract_features = ExtractionHelper.extract_features_LBP

# load the training dataset
train_path = "./KTH-texture-Data/train"
train_names = os.listdir(train_path)
# empty list to hold feature vectors and train labels
train_features = []
train_labels = [] #set()#[]
# load the test dataset
test_path = "./KTH-texture-Data/valid"
test_names = os.listdir(test_path)

# tempimg = cv2.imread("C:\\Users\\Noman\\VSCode Git\\IIPCV - Proj 2\\KTH-texture-Data\\train\\KTH_aluminium_foil\\1.jpg")
# graytempimg = cv2.cvtColor(tempimg,cv2.COLOR_BGR2GRAY)
# print(extract_features(graytempimg))
# exit()

# loop over the training dataset
print("[STATUS] Started feature extraction..")
for train_name in train_names:
        cur_path = train_path + "/" + train_name
        cur_label = train_name
        i = 1
        for file in glob.glob(cur_path + "/*.jpg"):
                # print("Processing Image - {} in {}".format(i, cur_label))
                ### read the training image
                image = cv2.imread(file)

                ### convert the image to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                ### extract haralick texture from the image
                features = extract_features(gray)
                if(isNormalizeFeaturesTrue):
                    features = features/255.0

                ### append the feature vector and label
                train_features.append(features)
                train_labels.append(cur_label)#.add(cur_label)#.append(cur_label)
                # print(cur_label)

                ### show loop update
                i += 1

# have a look at the size of our feature vector and labels
print("Training features: {}".format(np.array(train_features).shape))
print("Training labels: {}".format(np.array(train_labels).shape))

# create the classifier
print("[STATUS] Creating the classifier..")
clf_svm = LinearSVC(random_state=9,max_iter=10000,dual=False)

# fit the training data and labels
print("[STATUS] Fitting data/label to model..")
clf_svm.fit(train_features, train_labels)


# loop over the test images
print("[STATUS] Starting tests")
predictionsList = list()
validClassList = list()
for test_name in test_names:
        cur_path = test_path + "/" + test_name
        cur_label = test_name
        i = 1
        for file in glob.glob(cur_path + "/*.jpg"):
            # read the input image
            image = cv2.imread(file)

            # convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # extract haralick texture from the image
            features = extract_features(gray)

            # evaluate the model and predict label
            prediction = clf_svm.predict(features.reshape(1, -1))[0]
            predictionsList.append(prediction)
            validClassList.append(cur_label)
            if(isShowPredictionsEnabled):
                # show the label
                cv2.putText(image, prediction, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
                # display the output image
                cv2.imshow("Test_Image", image)
                cv2.waitKey(0)
print("Test predictions: {}".format(np.array(predictionsList).shape))
print("Test labels: {}".format(np.array(validClassList).shape))
print("Accuracy: " + str(accuracy_score(predictionsList, validClassList)))