import pandas
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

"""
Бинарная классификация отраженных сигналов радара
Цель
Реализовать классификацию между камнями (R) и  металлическими цилиндрами (M) на 
основе данных об отражении сигналов радара от поверхностей. 
60  входных  значений показывают  силу  отражаемого  сигнала  под определенным  углом. 
Входные данные нормализованы и находятся в промежутке от 0 до 1.
"""
dataframe = pandas.read_csv("sonar.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:60].astype(float)
Y = dataset[:, 60]

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

model = Sequential()
model.add(Dense(60, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history1 = model.fit(
    X, encoded_Y, epochs=100, batch_size=10, validation_split=0.1, verbose=None
)

plt.figure(figsize=(16, 4))
plt.subplot(2, 2, 1)
plt.title("loss relu 60, sigmoid 1")
plt.plot(history1.history["loss"], label="Loss", color="red")
plt.plot(history1.history["val_loss"], label="Val_loss", color="blue")
plt.legend()

plt.subplot(2, 2, 2)
plt.title("accuracy relu 60, sigmoid 1")
plt.plot(history1.history["accuracy"], label="Accuracy", color="red")
plt.plot(history1.history["val_accuracy"], label="Val_accuracy", color="blue")
plt.legend()

X_2 = dataset[:, 0:30].astype(float)
model_2 = Sequential()
model_2.add(Dense(60, activation="relu"))
model_2.add(Dense(1, activation="sigmoid"))
model_2.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history2 = model_2.fit(
    X_2, encoded_Y, epochs=100, batch_size=10, validation_split=0.1, verbose=None
)


plt.subplot(2, 2, 3)
plt.title("X=30 loss relu 60, sigmoid 1")
plt.plot(history2.history["loss"], label="Loss", color="red")
plt.plot(history2.history["val_loss"], label="Val_loss", color="blue")
plt.legend()

plt.subplot(2, 2, 4)
plt.title("accuracy relu 60, sigmoid 1")
plt.plot(history2.history["accuracy"], label="Accuracy", color="red")
plt.plot(history2.history["val_accuracy"], label="Val_accuracy", color="blue")
plt.legend()

model_3 = Sequential()
model_3.add(Dense(60, activation="relu"))
model_3.add(Dense(15, activation="relu"))
model_3.add(Dense(1, activation="sigmoid"))
model_3.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history3 = model_3.fit(
    X, encoded_Y, epochs=100, batch_size=10, validation_split=0.1, verbose=None
)


plt.figure(figsize=(16, 4))
plt.subplot(2, 2, 1)
plt.title("loss relu 60, relu 15, sigmoid 1")
plt.plot(history3.history["loss"], label="Loss", color="red")
plt.plot(history3.history["val_loss"], label="Val_loss", color="blue")
plt.legend()

plt.subplot(2, 2, 2)
plt.title("accuracy relu 60, relu 15, sigmoid 1")
plt.plot(history3.history["accuracy"], label="Accuracy", color="red")
plt.plot(history3.history["val_accuracy"], label="Val_accuracy", color="blue")
plt.legend()

model_4 = Sequential()
model_4.add(Dense(60, activation="relu"))
model_4.add(Dense(15, activation="relu"))
model_4.add(Dense(1, activation="sigmoid"))
model_4.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history4 = model_4.fit(
    X_2, encoded_Y, epochs=100, batch_size=10, validation_split=0.1, verbose=None
)


plt.subplot(2, 2, 3)
plt.title("X=30 loss relu 60, sigmoid 1")
plt.plot(history4.history["loss"], label="Loss", color="red")
plt.plot(history4.history["val_loss"], label="Val_loss", color="blue")
plt.legend()

plt.subplot(2, 2, 4)
plt.title("X=30 accuracy relu 60, sigmoid 1")
plt.plot(history4.history["accuracy"], label="Accuracy", color="red")
plt.plot(history4.history["val_accuracy"], label="Val_accuracy", color="blue")
plt.legend()

plt.show()


"""
Требования 
1. Изучить влияние кол-ва нейронов на слое на результат обучения модели. 
2. Изучить влияние кол-ва слоев на результат обучения модели 
3. Построить графики ошибки и точности в ходе обучения 
4. Провести сравнение полученных сетей, объяснить результат 
"""
