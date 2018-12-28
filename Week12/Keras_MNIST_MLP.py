from keras import layers, models
from keras import datasets
from keras.utils import np_utils
import matplotlib.pyplot as plt

def MLP_SEQ_FUNC(Nin, Nhid, Nout):
    model = models.Sequential()
    model.add(layers.Dense(Nhid, activation='sigmoid', input_shape=(Nin,)))
    model.add(layers.Dense(Nout, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def Data_FUNC():
    (X, y), (Xt, yt) = datasets.mnist.load_data()
    # class label change to one-hot
    Y = np_utils.to_categorical(y)
    Yt = np_utils.to_categorical(yt)
    L, W, H = X.shape
    # Vectorize
    X = X.reshape(-1, W*H)
    Xt = Xt.reshape(-1, W*H)
    # Normalization 0~255 to 0~1
    X = X / 255.0
    Xt = Xt / 255.0
    return (X, Y), (Xt, Yt)

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Test'], loc=0)

def plot_acc(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'], loc=0)

def main():
    Nin = 784
    Nhid = 100
    Nout = 10   # number of class

    myModel = MLP_SEQ_FUNC(Nin, Nhid, Nout)
    (X, Y), (Xt, Yt) = Data_FUNC()
    # Traning
    history = myModel.fit(X, Y, epochs=10, batch_size=100, validation_split=0.2)
    performance_test = myModel.evaluate(Xt, Yt, batch_size=100)
    print('Test Loss and Accuracy ->', performance_test)

    plot_loss(history)
    plt.show()
    plot_acc(history)
    plt.show()

# Run code
if __name__ == '__main__':
    main()