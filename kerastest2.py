import csv
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.model_selection import cross_val_score
#from sklearn.preprocessing import LabelEncoder
#from sklearn.model_selection import StratifiedKFold
#from sklearn.preprocessing import StandardScaler
#from sklearn.pipeline import Pipeline

##text=List of strings to be written to file
with open('myanswers.csv','w') as file:
    with open("test.csv", "r") as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            #print('line[{}] = {}'.format(i, line))
            if i > 0:
                file.write(line[0] + ',0\n')
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
#dataframe = pandas.read_csv("train_sample_10000.csv", header=1)
dataframe = pandas.read_csv("train_sample.csv", header=1)
dataset = dataframe.values
# split into input (X) and output (Y) variables
x_train = dataset[:,1:5].astype(float)
y_train = dataset[:,7]

# load dataset
#dataframe = pandas.read_csv("train_sample_only_ones.csv", header=1)
dataframe = pandas.read_csv("train_sample_10000.csv", header=1)
dataset = dataframe.values
# split into input (X) and output (Y) variables
x_test = dataset[:,1:5].astype(float)
y_test = dataset[:,7]

# load dataset
#dataframe = pandas.read_csv("train_sample_only_ones.csv", header=1)
dataframe = pandas.read_csv("test.csv", header=1)
dataset = dataframe.values
# split into input (X) and output (Y) variables
x_test_master = dataset[:,2:6].astype(float)

print(x_train)
print(y_train)

# create model
model = Sequential()
model.add(Dense(32, input_dim=4, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=128)
result = model.predict(x_test)
print(y_test)
print(result)

score = model.evaluate(x_test, y_test, batch_size=128)
print(score)


result = model.predict(x_test_master)
##text=List of strings to be written to file
with open('myanswers.csv','w') as file:
    with open("test.csv", "r") as f:
        reader = csv.reader(f, delimiter=",")
        for i, line in enumerate(reader):
            #print('line[{}] = {}'.format(i, line))
            if i > 0:
                file.write(line[0] + ',')

                if result[i-1][0] < .5:
                    file.write('0')
                else:
                    file.write('1')
                file.write('\n')