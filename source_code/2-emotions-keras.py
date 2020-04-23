import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as SC
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot
# import dataset
df = pd.read_csv("merge-mix.csv" , index_col=False)
df.loc[df['emotion'] == 3, 'emotion'] = 1 # 1 is happy
df.loc[df['emotion'] == 4, 'emotion'] = 0 # 0 is sad
X = df[[i for i in list(df.columns) if i != 'emotion']]
y = df['emotion']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 42)
sc = SC()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
# building our model
model = Sequential()
model.add(Dense(4, input_dim = X_train.shape[1])) 
model.add(Activation('relu'))
model.add(Dense(4))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(4))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
opt = 'adam'
model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])
# loss_and_metrics = model.evaluate(X_test, y_test, batch_size=64)
_, train_acc = model.evaluate(X_train, y_train, verbose=0)
_, test_acc = model.evaluate(X_test, y_test, verbose=0)
# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()
# display accuracy and confusion matrix
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.90)
cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred))