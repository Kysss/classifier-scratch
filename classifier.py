import os
import csv
import pandas as pd

from sklearn.feature_selection import RFE
import numpy as np
from sklearn.svm import SVC
from sklearn import svm
control_file = 'controldata.csv'
control_test_file = 'controltestdata.csv'
sz_file = 'szdata.csv'
sz_test_file = 'sztestdata.csv'




#Reading control data from a csv file
#Returns an array of arrays containing features of all subjects
def read_control_data ():
    alldata = pd.read_csv(control_file)
    #Drops the first column, which is the sequencial labling of subjects
    #and is not a feature
    #Axis=1 indicates column not row
    data = alldata.drop('Measure:volume' , axis = 1)
    return data.values

def set_control_labels( length ):
    control_labels = [0] * length
    return control_labels


def set_sz_labels( length ):
    sz_labels = [1] * length
    return sz_labels
    
#Reading schizophrenia data from a csv file
#Returns an array of arrays containing features of all subjects
def read_sz_data():
    alldata = pd.read_csv(sz_file)
    #Drops the first column, which is the sequencial labling of subjects
    #and is not a feature
    #Axis=1 indicates column not row
    data = alldata.drop('Measure:volume', axis = 1)
    return data.values

def read_control_test_data():
    alldata = pd.read_csv(control_test_file)
    data = alldata.drop('Measure:volume', axis =1)
    return data.values

def read_sz_test_data():
    alldata = pd.read_csv(sz_test_file)
    data = alldata.drop('Measure:volume', axis = 1)
    return data.values

def read_features():
    features = pd.read_csv(control_file)
    return features

def read_training_data():
    global training_x
    global training_y
    global feature_names
    feature_names = read_features().drop('Measure:volume', axis =1 ).columns.values
    c_data = read_control_data()
    c_label = set_control_labels(len(c_data))
    sz_data = read_sz_data()
    sz_label = set_sz_labels(len(sz_data))
    training_x = np.concatenate((c_data,sz_data), axis = 0)
    training_y = np.array(c_label + sz_label)

def read_testing_data():
    global testing_x
    global testing_y
    c_test_data = read_control_test_data()
    c_test_label = set_control_labels(len(c_test_data))
    sz_test_data = read_sz_test_data()
    sz_test_label = set_sz_labels(len(sz_test_data))
    testing_x = np.concatenate((c_test_data,sz_test_data), axis = 0)
    testing_y = np.array(c_test_label + sz_test_label)
    
def print_selected_feature_names ( selector_ranking ):
    features = []
    for i in range(len(selector_ranking)):
        if(selector_ranking[i] ==1):
            features.append(feature_names[i])
    return features
            
    
    
#print(c_data)
#print(sz_data)

#print(c_label)
#print(sz_label)

read_training_data()
#print(len(training_x))
#print(len(training_y))
read_testing_data()
#print(len(testing_x))
#print(len(testing_y))
print(feature_names)
print(len(feature_names))
clf = SVC()
clf.set_params(kernel='linear').fit(training_x,training_y)



X_test = testing_x

print(clf.predict(X_test))
print('without selector score')
print(clf.score(testing_x,testing_y))
'''print(clf.predict(training_x))
print(clf.score(training_x, training_y))'''


selectorlength = len(training_x)
selector = RFE(clf, selectorlength , step=1)
selector = selector.fit(testing_x, testing_y)
print('with selector')
print(selector.ranking_)
print('predicting with selector')
print(selector.predict(training_x))
print('selector score')
print(selector.score(training_x, training_y))
print(print_selected_feature_names(selector.ranking_))


linear = svm.LinearSVC()
linear.fit(training_x, training_y)
print(linear.predict(X_test))
print(linear.score(testing_x,testing_y))
'''print(linear.predict(training_x))
print(linear.score(training_x,training_y))'''





