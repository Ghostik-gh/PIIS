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
X = dataset[:, 0:4].astype(float)
Y = dataset[:, 4]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = to_categorical(encoded_Y)

model_1 = Sequential()
model_1.add(Dense(4, activation="relu"))
model_1.add(Dense(3, activation="softmax"))
model_1.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history_1 = model_1.fit(X, dummy_y, epochs=75, batch_size=10, validation_split=0.1)

model_2 = Sequential()
model_2.add(Dense(16, activation="relu"))
model_2.add(Dense(3, activation="softmax"))
model_2.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history_2 = model_2.fit(X, dummy_y, epochs=75, batch_size=10, validation_split=0.1)

model_3 = Sequential()
model_3.add(Dense(64, activation="relu"))
model_3.add(Dense(3, activation="softmax"))
model_3.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history_3 = model_3.fit(X, dummy_y, epochs=75, batch_size=10, validation_split=0.1)

model_4 = Sequential()
model_4.add(Dense(4, activation="relu"))
model_4.add(Dense(8, activation="relu"))
model_4.add(Dense(16, activation="relu"))
model_4.add(Dense(3, activation="softmax"))
model_4.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history_4 = model_4.fit(X, dummy_y, epochs=75, batch_size=10, validation_split=0.1)

model_5 = Sequential()
model_5.add(Dense(4, activation="relu"))
model_5.add(Dense(16, activation="linear"))
model_5.add(Dense(3, activation="softmax"))
model_5.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history_5 = model_5.fit(X, dummy_y, epochs=75, batch_size=10, validation_split=0.1)

model_6 = Sequential()
model_6.add(Dense(4, activation="relu"))
model_6.add(Dense(9, activation="softmax"))
model_6.add(Dense(3, activation="softmax"))
model_6.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history_6 = model_6.fit(X, dummy_y, epochs=75, batch_size=10, validation_split=0.1)

plt.figure(figsize=(16, 4))
plt.subplot(2, 3, 1)

plt.title("relu 4, softmax 3")
plt.plot(history_1.history["loss"], label="Loss", color="red")
plt.plot(history_1.history["val_loss"], label="Val_loss", color="blue")
plt.plot(history_1.history["accuracy"], label="Accuracy", color="green")
plt.plot(history_1.history["val_accuracy"], label="Val_accuracy", color="orange")
plt.legend()

plt.subplot(2, 3, 2)

plt.title("relu 16, softmax 3")
plt.plot(history_2.history["loss"], label="Loss", color="red")
plt.plot(history_2.history["val_loss"], label="Val_loss", color="blue")
plt.plot(history_2.history["accuracy"], label="Accuracy", color="green")
plt.plot(history_2.history["val_accuracy"], label="Val_accuracy", color="orange")
plt.legend()

plt.subplot(2, 3, 3)

plt.title("relu 64, softmax 3")
plt.plot(history_3.history["loss"], label="Loss", color="red")
plt.plot(history_3.history["val_loss"], label="Val_loss", color="blue")
plt.plot(history_3.history["accuracy"], label="Accuracy", color="green")
plt.plot(history_3.history["val_accuracy"], label="Val_accuracy", color="orange")
plt.legend()

plt.subplot(2, 3, 4)

plt.title("relu 4, relu 8, relu 16, softmax 3")
plt.plot(history_4.history["loss"], label="Loss", color="red")
plt.plot(history_4.history["val_loss"], label="Val_loss", color="blue")
plt.plot(history_4.history["accuracy"], label="Accuracy", color="green")
plt.plot(history_4.history["val_accuracy"], label="Val_accuracy", color="orange")
plt.legend()

plt.subplot(2, 3, 5)

plt.title("relu 4, linear 16, softmax 3")
plt.plot(history_5.history["loss"], label="Loss", color="red")
plt.plot(history_5.history["val_loss"], label="Val_loss", color="blue")
plt.plot(history_5.history["accuracy"], label="Accuracy", color="green")
plt.plot(history_5.history["val_accuracy"], label="Val_accuracy", color="orange")
plt.legend()

plt.subplot(2, 3, 6)

plt.title("relu 4, softmax 9, softmax 3")
plt.plot(history_6.history["loss"], label="Loss", color="red")
plt.plot(history_6.history["val_loss"], label="Val_loss", color="blue")
plt.plot(history_6.history["accuracy"], label="Accuracy", color="green")
plt.plot(history_6.history["val_accuracy"], label="Val_accuracy", color="orange")
plt.legend()


plt.show()


"""
Требования
1. Изучить различные архитектуры ИНС (Разное кол-во слоев, разное кол-во нейронов
на слоях)
2. Изучить обучение при различных параметрах обучения (параметры функции fit)
3. Построить графики ошибок и точности в ходе обучения
4. Выбрать наилучшую модель
"""
