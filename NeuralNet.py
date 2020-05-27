import numpy as np
import pandas as pd
import math

class neuralNet:
    def __init__(self, train):
        np.random.seed(1)
        # train refers to the training dataset
        # stepSize refers to the step size of gradient descent
        self.df = pd.read_csv(train)
        # read train.csv
        self.df.insert(0, 'X0', 1)
        # Insert a col of 1's at indx 0 with label X0
        self.nrows, self.ncols = self.df.shape[0], self.df.shape[1]
        # df.shape[0]= number of rows of a matrix (the first dimension)
        # df.shape[1]= number of col of a matrix (the second dimension)
        self.totavg, self.totstd_ev, self.totmax = [0] * self.ncols, [0] * self.ncols, [0] * self.ncols

        # df.iloc[:] the whole table
        # : means slice
        # df.iloc[:, 0:(self.ncols -1)] get the col 0-n-1
        self.df.dropna();
        cols = self.df.columns.values
        for i in range(1, self.ncols):
            tmp = self.df[cols[i]].values2
            self.totavg[i] = np.average(tmp)
            self.totstd_ev[i] = np.std(tmp)
            self.totmax[i] = max(tmp)
            new_tmp = [((j - self.totavg[i]) / self.totstd_ev[i]) / self.totmax[i] for j in tmp]
            self.df[cols[i]] = new_tmp
        self.X = self.df.iloc[:, 0:(self.ncols - 1)]
        self.y = self.df.iloc[:, (self.ncols - 1)]
        tmp = int(self.nrows * 0.8)
        self.TrainX = self.X.iloc[0:tmp].values.reshape(tmp, self.ncols - 1)
        self.Trainy = self.y.iloc[0:tmp].values.reshape(tmp, 1)
        self.TestX = self.X.iloc[tmp:].values.reshape(self.nrows - tmp, self.ncols - 1)
        self.Testy = self.y.iloc[tmp:].values.reshape(self.nrows - tmp, 1)
        # self.W = np.random.rand(self.ncols-1).reshape(self.ncols-1, 1)
        # numpy.random.rand create an array of thing
        self.weights1 = None
        self.weights2 = None
        self.outweight = None
        self.hlayer1 = None
        self.hlayer2 = None
        self.prediction = None
        self.activate = 0
        self.mse = None

    def buildNN(self, h1=3, h2=3):
        # hidden layer with default 3 units in each layer.
        # if number of units is less than 1, it will be automaticall set to 1
        # the max number of unit is up to 100, it will be automatically set to 100 if greater than 100
        if h1 < 1:
            h1 = 1
        if h2 < 1:
            h2 = 1
        if h1 > 100:
            h1 = 100
        if h2 > 100:
            h2 = 100
        self.weights1 = np.random.rand(self.TrainX.shape[1], h1)
        self.weights2 = np.random.rand(h1, h2)
        self.outweight = np.random.rand(h2, 1)

    def feedforward(self, activate=0):
        self.activate = activate
        # 0 is sigmoid
        # 1 is tanh
        # 2 is relu
        if activate != 0 and activate != 2 and activate != 1:
            print(
                "Invalid Activate Function Code. This Neural Network is going to use Sigmoid function as the activate function")
            activate = 0
        # np.seterr(over='ignore')
        if activate == 0:
            self.hlayer1 = sigmoid(np.dot(self.TrainX, self.weights1))
            self.hlayer2 = sigmoid(np.dot(self.hlayer1, self.weights2))
            self.prediction = sigmoid(np.dot(self.hlayer2, self.outweight))

        if activate == 1:
            self.hlayer1 = tanh(np.dot(self.TrainX, self.weights1))
            self.hlayer2 = tanh(np.dot(self.hlayer1, self.weights2))
            self.prediction = tanh(np.dot(self.hlayer2, self.outweight))

        if activate == 2:
            # Because Relu can be only applyed to the hidden layer, the output layer will use sigmoid function
            self.hlayer1 = relu(np.dot(self.TrainX, self.weights1))
            self.hlayer2 = relu(np.dot(self.hlayer1, self.weights2))
            self.prediction = sigmoid(np.dot(self.hlayer2, self.outweight))

    def backpropagation(self, rate=0.3):
        # error: 1/2 sum(t-o)^2
        def driv_sig(var):
            var = sigmoid(-var)
            return var * (1 - var)

        def driv_tanh(var):
            var = tanh(var)  # in case of vary small number will cause devide 0
            return 1 - pow(var, 2)

        def driv_relu(var):
            ret = var
            for i in range(0, var.shape[0]):
                for j in range(0, var.shape[1]):
                    if var[i][j] > 0:
                        ret[i][j] = 1
                    if var[i][j] < 0:
                        ret[i][j] = 0
            return ret

        if self.activate == 0:
            error = self.Trainy - self.prediction
            mse = (1 / (2 * self.Trainy.shape[0])) * np.dot(error.T, error)
            self.mse = mse
            l3_error = self.Trainy - self.prediction
            dif_outweight = l3_error * driv_sig(self.prediction)
            l2_error = np.dot(dif_outweight, self.outweight.T)
            dif_h2weight = l2_error * driv_sig(self.hlayer2)
            l1_error = np.dot(dif_h2weight, self.weights2.T)
            dif_h1weight = l1_error * driv_sig(self.hlayer1)
            self.outweight += rate * self.hlayer2.T.dot(dif_outweight)
            self.weights2 += rate * self.hlayer1.T.dot(dif_h2weight)
            self.weights1 += rate * self.TrainX.T.dot(dif_h1weight)

        if self.activate == 1:
            error = self.Trainy - self.prediction
            mse = (1 / (2 * self.Trainy.shape[0])) * np.dot(error.T, error)
            self.mse = mse
            l3_error = self.Trainy - self.prediction
            dif_outweight = l3_error * driv_tanh(self.prediction)
            l2_error = np.dot(dif_outweight, self.outweight.T)
            dif_h2weight = l2_error * driv_tanh(self.hlayer2)
            l1_error = np.dot(dif_h2weight, self.weights2.T)
            dif_h1weight = l1_error * driv_tanh(self.hlayer1)
            self.outweight += rate * self.hlayer2.T.dot(dif_outweight)
            self.weights2 += rate * self.hlayer1.T.dot(dif_h2weight)
            self.weights1 += rate * self.TrainX.T.dot(dif_h1weight)

        if self.activate == 2:
            error = self.Trainy - self.prediction
            mse = (1 / (2 * self.Trainy.shape[0])) * np.dot(error.T, error)
            self.mse = mse
            l3_error = self.Trainy - self.prediction
            dif_outweight = l3_error * driv_sig(self.prediction)
            l2_error = np.dot(dif_outweight, self.outweight.T)
            dif_h2weight = l2_error * driv_relu(self.hlayer2)
            l1_error = np.dot(dif_h2weight, self.weights2.T)
            dif_h1weight = l1_error * driv_relu(self.hlayer1)
            self.outweight += rate * self.hlayer2.T.dot(dif_outweight)
            self.weights2 += rate * self.hlayer1.T.dot(dif_h2weight)
            self.weights1 += rate * self.TrainX.T.dot(dif_h1weight)

    def Predict(self):
        if self.activate == 0:
            P1 = sigmoid(np.dot(self.TestX, self.weights1))
            P2 = sigmoid(np.dot(P1, self.weights2))
            Prediction = sigmoid(np.dot(P2, self.outweight))
            error = self.Testy - Prediction
        if self.activate == 1:
            P1 = tanh(np.dot(self.TestX, self.weights1))
            P2 = tanh(np.dot(P1, self.weights2))
            Prediction = tanh(np.dot(P2, self.outweight))
            error = self.Testy - Prediction
        if self.activate == 2:
            P1 = relu(np.dot(self.TestX, self.weights1))
            P2 = relu(np.dot(P1, self.weights2))
            Prediction = sigmoid(np.dot(P2, self.outweight))
            error = self.Testy - Prediction

        mse = (1 / (2 * self.Testy.shape[0])) * np.dot(error.T, error)
        print("MSE on Test data set: {}".format(mse))
        print("Weight1")
        print(self.weights1)
        print("Weight2")
        print(self.weights2)
        print("Weight3")
        print(self.outweight)


def sigmoid(var):
    var = np.clip(var, -500, 500)
    return 1 / (1 + np.exp(-var))


def tanh(var):
    var = np.clip(var, -500, 500)
    etox = pow(math.exp(1), var)
    etonx = pow(math.exp(1), -var)
    return (etox - etonx) / (etox + etonx)


def relu(var):
    return np.maximum(0, var)


if __name__ == "__main__":
    model = neuralNet("dataSets.csv")
    model.buildNN(3, 3)
    epochs = 100
    act = 0
    try:
        act=int(input("Please Select the Activate function you Like\n0. Sigmoid\n1. Tanh\n2. Relu\n"))
    except ValueError:
        print("Cannot recognize the Activate Function\nUsing default Activate Function...")
        act = 0
    if not(act==0 or act == 1 or act ==2):
        print("Cannot recognize the Activate Function\nUsing default Activate Function...")
        act =0
    # print(model.Trainy)
    for i in range(epochs):
        model.feedforward(act)
        model.backpropagation(0.0005)
        if i % 10 == 0:
            print("MSE on Train Data Sets: {} ".format(model.mse))
            print("========================")
    model.Predict()
