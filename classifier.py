import os
import csv
import pandas as pd

import numpy as np
from sklearn.svm import SVC

control_file = 'controldata.csv'
sz_file = 'szdata.csv'




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


c_data = read_control_data()
c_label = set_control_labels(len(c_data))
sz_data = read_sz_data()
sz_label = set_sz_labels(len(sz_data))

#print(c_data)
#print(sz_data)

#print(c_label)
#print(sz_label)


training_x = np.concatenate((c_data,sz_data), axis = 0)
training_y = np.array(c_label + sz_label)
print(len(training_x))
print(len(training_y))
clf = SVC()
clf.set_params(kernel='linear').fit(training_x,training_y)
X_test = sz_data
print(clf.predict(X_test))




