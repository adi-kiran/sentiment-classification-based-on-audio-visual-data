import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as SC
import numpy as np
from matplotlib import pyplot
df = pd.read_csv("All-emotions.csv")
# THe dataset contains numbers for the truth value , replacing it with the associated emotion for that number.
df.loc[df['emotion'] == 1, 'emotion'] = "neutral"
df.loc[df['emotion'] == 2, 'emotion'] = "calm"
df.loc[df['emotion'] == 3, 'emotion'] = "happy"
df.loc[df['emotion'] == 4, 'emotion'] = "sad"
df.loc[df['emotion'] == 5, 'emotion'] = "angry"
df.loc[df['emotion'] == 6, 'emotion'] = "fearful"
df.loc[df['emotion'] == 7, 'emotion'] = "disgust"
df.loc[df['emotion'] == 8, 'emotion'] = "surprised"
# loading data
x = df[[i for i in list(df.columns) if i != 'emotion']]
y = df['emotion']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.10, random_state = 42)
sc = SC()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
X_train, X_test = np.array(X_train), np.array(X_test)
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y = encoder.transform(y_train)
# convert integers to dummy variables (i.e. one hot encoded)
y_train = np_utils.to_categorical(encoded_Y)
encoder.fit(y_test)
encoded_Y = encoder.transform(y_test)
# convert integers to dummy variables (i.e. one hot encoded)
y_test = np_utils.to_categorical(encoded_Y)
# build model
model = Sequential()
model.add(Dense(32, input_dim=156, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
history = model.fit(X_train, y_train, validation_split=0.33, epochs=100, batch_size=200)
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
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))