import numpy as np


def ReLU(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    # softmax(i) = e^xi / sum(e^xj for all j)
    exp = np.exp(x)
    return exp / np.sum(exp)


def cse(y, out):
    # L = -sum(y_true * log(y_pred))
    eps = 1e-8
    return -np.sum(y * np.log(out + eps))


class TwoLayerMLP:
    # 2-layer MLP: input size 4, hidden size 8, output size 3

    def __init__(self, input_size=4, hidden_size=8, out_size=3):
        self.W1 = np.random.randn(hidden_size, input_size)
        self.W2 = np.random.randn(out_size, hidden_size)
        self.b1 = np.random.randn(hidden_size)
        self.b2 = np.random.randn(out_size)

        for param in ["W1", "W2", "b1", "b2"]:
            setattr(self, f"m{param}", np.zeros_like(getattr(self, param)))
            setattr(self, f"v{param}", np.zeros_like(getattr(self, param)))

    def forward(self, x):
        self.L1 = self.W1 @ x + self.b1
        self.ReLU1 = ReLU(self.L1)
        self.L2 = self.W2 @ self.ReLU1 + self.b2

        out = softmax(self.L2)

        return out

    def loss(self, out, y):
        return -np.sum(y * np.log(out))

    def backward(self, out, y, x):
        # L = -sum(y * log(out))
        # L = y * log(out)
        # L = y * log(softmax(L2))
        # DL/DL2 = out - y = f'
        # L = f(L2)

        # dL/dL2 = out - y

        # L = f(W2 * Relu(L1) + b2))

        # dL / W2 = (out - y) * Relu(L1) ?

        # dL / B2 = (out - y)

        # dL / dL1 = (out - y) * W2 * L1 > 0 ?

        # dl / dW1 = (out - y) * W2 * L1 > 0 * x

        delta = out - y
        # print(delta)
        # print(self.ReLU1.shape)
        self.dW2 = np.outer(delta, self.ReLU1)
        # print(self.dW2.shape)
        self.db2 = delta
        # print(delta.shape)
        dL1 = (delta @ self.W2) * (self.L1 > 0)
        self.dW1 = np.outer(dL1, x)
        # print(self.dW1.shape, self.W1.shape)
        # print()
        self.db1 = dL1

        # self.dW1 =
        #
        # self.db1 =
        # self.db2 =

    def sgd_optim(self):
        self.W1 = self.W1 - self.lr * self.dW1
        self.W2 = self.W2 - self.lr * self.dW2
        self.b1 = self.b1 - self.lr * self.db1
        self.b2 = self.b2 - self.lr * self.db2

    def adam_optim(self):
        """
        m = beta1 * m + (1 - beta1) * grad        # momentum (first moment)
        v = beta2 * v + (1 - beta2) * grad²       # gradient squared (second moment)
        w = w - lr * m / (sqrt(v) + epsilon)
        """

        def adam(w, dw, m, v):
            m[:] = self.beta1 * m + (1 - self.beta1) * dw
            v[:] = self.beta2 * v + (1 - self.beta2) * dw**2
            return w - self.lr * m / (np.sqrt(v) + self.epsilon)

        self.W1 = adam(self.W1, self.dW1, self.mW1, self.vW1)
        self.W2 = adam(self.W2, self.dW2, self.mW2, self.vW2)
        self.b1 = adam(self.b1, self.db1, self.mb1, self.vb1)
        self.b2 = adam(self.b2, self.db2, self.mb2, self.vb2)

    def train(self, data, learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = 1e-8

        x = data["input"]
        y = data["output"]

        assert x.shape[0] == y.shape[0]

        for i in range(x.shape[0]):
            out = self.forward(x[i])
            loss = self.loss(out, y[i])
            print(loss)
            self.backward(out, y[i], x[i])
            self.adam_optim()


n_iter = 15
data = {
    "input": np.random.randn(n_iter, 4),
    "output": np.random.dirichlet(np.ones(3), n_iter),
}

x = np.random.randn(4)
data = {"input": np.tile(x, (100, 1)), "output": np.tile(np.array([1, 0, 0]), (100, 1))}

net = TwoLayerMLP(input_size=data["input"].shape[1], out_size=data["output"].shape[1])

net.train(data)

# net.train()
