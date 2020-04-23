import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as SC
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import dataset
df = pd.read_csv("merge-mix.csv" , index_col=False)
# select columns needed for x and y
hello = list(df.columns)
hello.remove('emotion')
X = df[[i for i in list(hello)]]
y = df['emotion']
# one hot encoding the y (results in 8 columns for each of the eight emotions)
y = pd.get_dummies(y,prefix='emotion')
# creating train and test data by taking 70/30 split, and scaling the values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 1)
sc = SC()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
tf.reset_default_graph()
# Defining the number of neurons in each of the layers
# We are using a Deep neural network with three hidden layers
size_input = 156  # 156 inputs
size_output = 2   # 8 emotions, hence 8 output layer neurons
size_hidden1 = 8 # hidden layer 1
size_hidden2 = 8 # hidden layer 2
size_hidden3 = 4 # hidden layer 3
# Defining the parameters for the model
training_epochs = 20
display_step = 10
batch_size = 200
# building the model
x = tf.placeholder(tf.dtypes.float64,name="x")
y = tf.placeholder(tf.dtypes.float64,name="y")
keep_prob = tf.placeholder(tf.dtypes.float64)
# first hidden layer's weights and biases
Wh1 = tf.get_variable(shape=[size_input, size_hidden1] , dtype = tf.dtypes.float64, name="Wh1")
bh1 = tf.get_variable(shape=[1, size_hidden1], dtype = tf.dtypes.float64, name="bh1")
# second hidden layer's weights and biases
Wh2 = tf.get_variable(shape=[size_hidden1, size_hidden2], dtype = tf.dtypes.float64, name="Wh2")
bh2 = tf.get_variable(shape=[1, size_hidden2], dtype = tf.dtypes.float64, name="bh2")
# third hidden layer's weights and biases
Wh3 = tf.get_variable(shape=[size_hidden2, size_hidden3], dtype = tf.dtypes.float64, name="Wh3")
bh3 = tf.get_variable(shape=[1, size_hidden3], dtype = tf.dtypes.float64, name="bh3")
# output layer's weights and biases
Wy = tf.get_variable(shape=[size_hidden3, size_output], dtype = tf.dtypes.float64, name="Wy")
by = tf.get_variable(shape=[1, size_output], dtype = tf.dtypes.float64, name="by")
# computing the result for the first hidden layer
a1 = tf.add(tf.matmul(x,Wh1),bh1)
lh1 = tf.nn.relu(a1)
# computing the result for the second hidden layer
a2 = tf.add(tf.matmul(lh1,Wh2),bh2)
lh2 = tf.nn.relu(a2)
# computing the result for the third hidden layer
a3 = tf.matmul(lh2,Wh3) + bh3
lh3 = tf.nn.relu(a3)
# computing the result for the output
pred = tf.add(tf.matmul(lh3,Wy) , by, name="predictions")
# appling softmax for classification
# pred holds the final predictions
# pred = tf.nn.softmax(a4,name="predictions")
# cost function being used is categorical cross entropy
# cost = tf.reduce_mean(tf.compat.v1.keras.losses.categorical_crossentropy(y, pred))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred))
# optimizer being used is AdamOptimizer 
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
# calculating correct predictions and accuracy from validation set
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1),name="correct_pred")
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="accuracy")
# the training of the network
saver = tf.train.Saver()
costs = []
accs = []
final_pred = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0.0
        total_batch = int(len(X_train) / batch_size)
        x_batches = np.array_split(X_train, total_batch)
        y_batches = np.array_split(y_train, total_batch)
        for i in range(total_batch):
            batch_x, batch_y = x_batches[i], y_batches[i]
            _, c = sess.run([optimizer, cost], feed_dict={ x: batch_x, y: batch_y, keep_prob: 0.8})
            avg_cost += c
        avg_cost/=total_batch
        costs.append(avg_cost)
        accs.append(acc.eval({x: batch_x, y: batch_y, keep_prob: 1.0}))
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
    print("Optimization Finished!")
    final_pred.extend(sess.run(pred,feed_dict={x: X_test, y: y_test, keep_prob: 1.0}))
    print("Accuracy:", acc.eval({x: X_test, y: y_test, keep_prob: 1.0}))
    saved_loc = saver.save(sess,"saved_models/2_emotion_model.ckpt")
# we save the trained weights into the saved_models folder
# it can be loaded and used for predictions later on
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score
x = []
y = []
for i,j in zip(final_pred,y_test):
  i = list(i)
  j = list(j)
  x.append(i.index(max(i))+1)
  y.append(j.index(max(j))+1)
print("Accuracy", accuracy_score(x,y))
print("Precision", precision_score(x,y,average='macro'))
print("Recall", recall_score(x,y,average='macro'))
print("Confusion Matrix\n", confusion_matrix(x,y))
from matplotlib import pyplot
# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(costs, label='train')
pyplot.legend()
# plot accuracy during training
pyplot.subplot(212)
pyplot.show()
pyplot.subplot(211)
pyplot.title('Accuracy')
pyplot.plot(accs, label='train')
pyplot.legend()
pyplot.show()