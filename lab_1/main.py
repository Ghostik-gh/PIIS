import pandas
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

"""
Цель
Реализовать классификацию сортов растения ирис (Iris Setosa - 0, Iris Versicolour - 1, Iris
Virginica - 2) по четырем признакам: размерам пестиков и тычинок его цветков.
"""

dataframe = pandas.read_csv("iris.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = to_categorical(encoded_Y)

model = Sequential()

model.add(Dense(4, activation="relu"))
model.add(Dense(3, activation="softmax"))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X, dummy_y, epochs=75, batch_size=10, validation_split=0.1)

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.plot(history.history['accuracy'], label='accuracy:')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('График ошибок и точности обучения')
plt.ylabel('loss & accuracy')
plt.xlabel('epoch')
plt.legend(loc="upper left")
plt.show()



"""
Требования
1. Изучить различные архитектуры ИНС (Разное кол-во слоев, разное кол-во нейронов
на слоях)
2. Изучить обучение при различных параметрах обучения (параметры функции fit)
3. Построить графики ошибок и точности в ходе обучения
4. Выбрать наилучшую модель
"""