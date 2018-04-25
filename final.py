#!/usr/bin/env python3
"""
Module Docstring
"""
import csv
#import sys
import pandas as pd
import numpy as np
import pydot
import tensorflow as tf
import graphviz
from keras.models import Sequential
#from keras.utils import plot_model
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import keras.backend as K
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc


__author__ = "Danny Choi"
__version__ = "0.1.0"
__license__ = "MIT"

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value

def preprocessTraining(df):
    #preprocess data
    df['time_month'] = df.click_time.str[5:7]
    df['time_day']   = df.click_time.str[8:10]
    df['time_hr']    = df.click_time.str[11:13]
    df['time_min']   = df.click_time.str[14:16]
    df['time_sec']   = df.click_time.str[17:20]
    df = df.drop(['ip', 'attributed_time', 'click_time'], axis = 1)

    # split into input (X) and output (Y) variables
    x_data = df.drop(['is_attributed'], axis = 1).values
    y_data = df.drop(['app','device','os','channel','time_month','time_day','time_hr','time_min','time_sec'], axis = 1).values
    return (x_data, y_data)

def preprocessTest(df):
    #preprocess data
    df['time_month'] = df.click_time.str[5:7]
    df['time_day']   = df.click_time.str[8:10]
    df['time_hr']    = df.click_time.str[11:13]
    df['time_min']   = df.click_time.str[14:16]
    df['time_sec']   = df.click_time.str[17:20]
    x_data = df.drop(['click_id', 'ip', 'click_time'], axis = 1).values
    return (x_data)

def main():
    """ Main entry point of the app """

    #I/O files
    masterTrainingData = "train_sample.csv"
    masterTestData = "test.csv"
    sampleTrainingData = "train_sample_10000.csv"
    submissionTemplate = "sample_submission.csv"
    submissionOutput = "mySubmission.csv"
    
    # fix random seed for reproducibility
    seed = 69
    np.random.seed(seed)

    #load training data from csv
    dataframe = pd.read_csv(masterTrainingData, header=0)

    # split into input (X) and output (Y) variables
    x_train_master, y_train_master = preprocessTraining(dataframe);

    #print(x_train)
    #print(len(x_train))
    #print(y_train)
    
    #downsample to avoid unbalanced data
    dataframe_train_neg = dataframe[(dataframe['is_attributed'] == 0)]
    dataframe_train_pos = dataframe[(dataframe['is_attributed'] == 1)]
    print(len(dataframe_train_neg))
    print(len(dataframe_train_pos))

    dataframe_train_neg_sample = dataframe_train_neg.sample(n=4000)
    dataframe_train_pos_sample = dataframe_train_pos.sample(n=227)
    
    print(len(dataframe_train_neg_sample))
    print(len(dataframe_train_pos_sample))

    dataframe_train_comb = pd.concat([dataframe_train_neg_sample,dataframe_train_pos_sample])
    x_train, y_train = preprocessTraining(dataframe_train_comb)
    
    #submission = pd.read_csv(submissionTemplate)
    #submission['is_attributed'] = y_predss
    #submission.to_csv(submissionOutput, index=False)
    #print(submission.head())

    # create model
    model = Sequential()
    model.add(Dense(6, input_dim=9, kernel_initializer='normal', activation='relu'))
    model.add(Dense(3, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='tanh'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=8, batch_size=128)
    results = model.predict(x_train)
    results = np.where(results > 0.5, 1, 0)
    #plot_model(model, to_file='model.png')

    negCount = 0
    posCount = 0
    for i in range(0, len(results)):
        if results[i][0] == 1:
            posCount += 1
        else:
            negCount += 1
    print(negCount)
    print(posCount)
    
    score = model.evaluate(x_train_master, y_train_master, batch_size=128)
    print(score)
    
    #test on full training data
    results = model.predict(x_train_master)
    results = np.where(results > 0.5, 1, 0)
    
    negCount = 0
    posCount = 0
    for i in range(0, len(results)):
        if results[i][0] == 1:
            posCount += 1
        else:
            negCount += 1
    print(negCount)
    print(posCount)
    false_positive_rate, recall, thresholds = roc_curve(y_train_master, results)
    roc_auc = auc(false_positive_rate, recall)
    print(roc_auc)
    
    #dataframe = pd.read_csv(masterTestData, header=0)
    #x_test = preprocessTest(dataframe)

    #myPredictions = model.predict(x_test)
    #myPredictions = np.where(myPredictions > 0.5, 1, 0)

    #output to a submission file
    #mySubmission = pd.read_csv(submissionTemplate)
    #mySubmission['is_attributed'] = myPredictions
    #mySubmission.to_csv('mySubmission.csv', index=False)
    #print(mySubmission.head())


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
