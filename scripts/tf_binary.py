from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as SC
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Reading the CSV file
df = pd.read_csv("cleaned_binary.csv" , index_col=False)
cols = df.columns
df.drop(columns=['Unnamed: 0'] , inplace=True)

df = df.sample(n=5000 , replace=False)

# X = The feaures of the dataset. Basically all the columns in the dataset except the label or the 'emotion column'
# y = The label or the truth value column.
X = df[[i for i in list(df.columns) if i != 'emotion']]
y = df['emotion']

# Splitting the data into training and testing set. Split ratio is 80 training and 20 testing.
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

# To scale and centre the data

sc = SC()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Converting the data into numpy arrays.
X_train, Y_train, X_test, Y_test = np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)

# Defining the learning rate and the number of epochs
lr = 0.01
epochs = 50

# Defining the placeholders to feed our input to the model
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Defining the model parameters before building.
size_input = 156
size_output = 1
size_hidden1 = 8
size_hidden2 = 8
size_hidden3 = 1

Wh1 = tf.Variable(tf.random_normal([size_input, size_hidden1]))
bh1 = tf.Variable(tf.random_normal([1, size_hidden1]))

Wh2 = tf.Variable(tf.random_normal([size_hidden1, size_hidden2]))
bh2 = tf.Variable(tf.random_normal([1, size_hidden2]))

Wh3 = tf.Variable(tf.random_normal([size_hidden2, size_hidden3]))
bh3 = tf.Variable(tf.random_normal([1, size_hidden3]))

Wy = tf.Variable(tf.random_normal([size_hidden3, size_output]))
by = tf.Variable(tf.random_normal([1, size_output])) 

a1 = tf.matmul(X,Wh1) + bh1
lh1 = tf.nn.relu(a1)

a2 = tf.matmul(lh1,Wh2) + bh2
lh2 = tf.nn.relu(a2)

a3 = tf.matmul(lh2,Wh3) + bh3
lh3 = tf.nn.sigmoid(a3)

pred = tf.matmul(lh3,Wy) + by

loss = tf.compat.v1.keras.losses.binary_crossentropy(Y_train, pred)
loss = tf.reduce_sum(loss)
opt = tf.compat.v1.train.AdamOptimizer(learning_rate=lr).minimize(loss)

init = tf.global_variables_initializer()

#Run
with tf.Session() as sess:
    sess.run(init)
    
    for i in range(epochs):
        count = 0
        while count < 16:
            sess.run(opt, feed_dict={X:X_train, Y:Y_train})
            count = count + 1
        if i % 1 == 0:
            print("Loss:", loss.eval({X:X_train, Y:Y_train}))
            
    # Calculating the metrics
    y_pred=sess.run(pred, feed_dict={X:X_test})
    y_pred = (y_pred > 0.90)

    cm = confusion_matrix(Y_test, y_pred)
    print(cm)
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(Y_test, y_pred)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(Y_test, y_pred)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(Y_test, y_pred)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(Y_test, y_pred)
    print('F1 score: %f' % f1)

