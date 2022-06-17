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

data_index = ['A','B','C','D','E','F','G','H','I','J'] # 读取的列
vr_weight = [i+1 for i in range (10)] #权重

"""处理数据集"""
def dataset():
    df = pd.read_excel('./dataset.xlsx',usecols=(data_index))
    # print(df.shape)
    score = df.dot(vr_weight) #计算加权得分
    # print(score)
    # df['score'] = score
    # print(df)
    score = np.array(score)
    return df,score
    
def load_data(data,score):
    docX, docY = [], []
    for i in range(len(data)):
        docX.append(data.iloc[i].values)
        docY.append(score[i])
    alsX = np.array(docX)
    alsY = np.array(docY)
    return alsX, alsY


def train_test_split(df,score,test_size):
    ntrn = round(len(df) * (1 - test_size))
    ntrn = int(ntrn)
    X_train, y_train = load_data(df.iloc[0:ntrn],score[0:ntrn])
    X_test, y_test = load_data(df.iloc[ntrn:],score[ntrn:])
    return (X_train, y_train), (X_test, y_test)

def train(X_train,y_train):
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    # tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
              loss="mean_squared_error",
              metrics=['sparse_categorical_accuracy'])

    checkpoint_save_path = "./checkpoint/mnist.ckpt"
    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        model.load_weights(checkpoint_save_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                    save_weights_only=True,
                                                    save_best_only=True)

    history = model.fit(X_train, y_train, batch_size=5, epochs=15, validation_split=0.05, validation_freq=1,
                        callbacks=[cp_callback])
    model.summary()
    print(model.trainable_variables)
    file = open('./weights.txt', 'w')
    for v in model.trainable_variables:
        file.write(str(v.name) + '\n')
        file.write(str(v.shape) + '\n')
        file.write(str(v.numpy()) + '\n')
    file.close()

    model.save('saved_model/my_model')

    # 显示训练集和验证集的acc和loss曲线
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def predict(X_test,y_test):
    model = tf.keras.models.load_model('saved_model/my_model')
    predicted = model.predict(X_test)
    dataf = pd.DataFrame(predicted[:])
    # dataf.columns = ["predict"]
    # dataf["input"] = y_test[:]
    dataf.plot(figsize=(18, 5))
    plt.xlabel("x")
    # plt.xlabel("timr/s")
    plt.ylabel("y")
    plt.title("with time ")
    plt.show()


def main():
    df,score = dataset()
    (X_train, y_train), (X_test, y_test) = train_test_split(df[data_index],score,0.1)
    print(X_train)
    print(y_train)
    print(X_test)
    print(y_test)

    train(X_train,y_train)
    # while True:

    predict(X_test,y_test)
# if __name__ == '__name__':
main()