# tensorflow_version 2.0
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import os 

plt.rcParams['font.sans-serif'] = ['SimHei'] #运行配置参数中的字体（font）为黑体（SimHei）
plt.rcParams['axes.unicode_minus'] = False #运行配置参数总的轴（axes）正常显示正负号（minus）

in_out_neurons = 1
hidden_neurons = 300
length_of_sequences = 100

def load_data(data, n_prev = 100): 
	docX, docY = [], []
	for i in range(len(data)-n_prev):
		docX.append(data.iloc[i:i+n_prev].values)
		docY.append(data.iloc[i+n_prev].values)
	alsX = np.array(docX)
	alsY = np.array(docY)
	return alsX, alsY

# 划分出90%的数据用于训练
def train_test_split(df, test_size = 0.1, n_prev = 100):
	ntrn = round(len(df) * (1 - test_size))
	ntrn = int(ntrn)
	X_train, y_train = load_data(df.iloc[0:ntrn], n_prev)
	X_test, y_test = load_data(df.iloc[ntrn:], n_prev)
	return (X_train, y_train), (X_test, y_test)


def train(X_train,y_train):
    model = Sequential()  
    model.add(tf.keras.layers.LSTM(hidden_neurons, batch_input_shape=(None, length_of_sequences, in_out_neurons), return_sequences=False))  
    model.add(Dense(1))  
    model.add(Activation("linear"))  
    model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(0.001))

    checkpoint_save_path = "./checkpoint/LSTM.ckpt"

    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        model.load_weights(checkpoint_save_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    monitor='val_loss')

                                                    
    history = model.fit(X_train, y_train, batch_size=600, epochs=5, validation_split=0.05 , callbacks=[cp_callback])
    model.summary()

    file = open('./weights.txt', 'w')  # 参数提取
    for v in model.trainable_variables:
        file.write(str(v.name) + '\n')
        file.write(str(v.shape) + '\n')
        file.write(str(v.numpy()) + '\n')
    file.close()

    model.save('saved_model/my_model')

    # # 显示训练集和验证集的acc和loss曲线
    # acc = history.history['mean_squared_error']
    # val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # plt.subplot(1, 2, 1)
    # plt.plot(acc, label='Training Accuracy')
    # plt.plot(val_acc, label='Validation Accuracy')
    # plt.title('Training and Validation Accuracy')
    # plt.legend()

    # plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


# def eval():
#     return NULL

def predict(X_test,y_test):
    model = tf.keras.models.load_model('saved_model/my_model')
    predicted = model.predict(X_test)
    dataf = pd.DataFrame(predicted[:])
    dataf.columns = ["predict"]
    dataf["input"] = y_test[:]
    dataf.plot(figsize=(18, 5))
    plt.xlabel("时间/s")
    # plt.xlabel("timr/s")
    plt.ylabel("pressure/pa")
    plt.title("pa with time ")
    plt.show()

steps_per_cycle = 100 # T
number_of_cycles = 80
def myfunc(x):
    y = np.sin(x * (2 * np.pi / steps_per_cycle)) + random.uniform(-1.0, +1.0) * 0.75 
    return y

def main():
    # 在“t”列存放1、2、3……50*80+1
    df = pd.DataFrame(np.arange(steps_per_cycle * number_of_cycles + 1), columns=["t"])
    # 在“pa”列存放sin(t+噪声)
    df["pa"] = df.t.apply(lambda x: myfunc(x))
    # 以“t”为横轴，“pa”为纵轴绘图
    df[["pa"]].head(steps_per_cycle * 10).plot()
    plt.show()

    (X_train, y_train), (X_test, y_test) = train_test_split(df[["pa"]], n_prev = length_of_sequences)

    train(X_train,y_train)
    # while True:

    predict(X_test,y_test)

# if __name__ == '__name__':
main()