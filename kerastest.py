#!/usr/bin/env python3
"""
Module Docstring
"""
import csv
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense


__author__ = "Danny Choi"
__version__ = "0.1.0"
__license__ = "MIT"


def main():
    """ Main entry point of the app """
    csv1 = np.genfromtxt('train_sample_10000.csv', dtype=float, delimiter=",", skip_header=1)
    manual_y_train = []
    #start
    with open("train_sample_10000.csv", "r") as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            #print('line[{}] = {}'.format(i, line))
            if i > 0:
                temp_y_train_array = [line[7]]
                manual_y_train.append(temp_y_train_array)

    #end
    print(manual_y_train)
    #csv = np.genfromtxt('train_sample.csv', delimiter=",", skip_header=1)
    x_train = csv1[:,1:5]
    y_train = csv1[:,7]
    csv2 = np.genfromtxt('train_sample_20000.csv', delimiter=",", skip_header=1)
    x_test = csv2[:,1:5]
    y_test = csv2[:,7]
    print(len(x_train), 'train sequences')
    print(len(y_train), 'test sequences')
    print(x_train)
    print(y_train)
    model = Sequential()
    model.add(Dense(units=32, activation='relu', input_dim=4))
    model.add(Dense(units=2, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    model.fit(x_train, np.array(manual_y_train), epochs=5, batch_size=128)
    result = model.predict(x_test)
    print(result)

if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
