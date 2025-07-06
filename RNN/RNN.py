import numpy as np

class RNN:
    def __init__(self, input_size, hidden_size, output_size, seq_len):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.seq_len = seq_len

        # 가중치 초기화
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01  # input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
        self.Why = np.random.randn(output_size, hidden_size) * 0.01  # hidden to output
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, inputs):
        self.inputs = inputs  # list of input vectors
        self.hs = {}
        self.hs[-1] = np.zeros((self.hidden_size, 1))
        self.outputs = {}

        for t in range(len(inputs)):
            x = np.reshape(inputs[t], (self.input_size, 1))
            self.hs[t] = self.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, self.hs[t-1]) + self.bh)
            self.outputs[t] = np.dot(self.Why, self.hs[t]) + self.by
        return self.outputs

    def backward(self, doutputs, learning_rate=0.01):
        # 초기화
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        dh_next = np.zeros((self.hidden_size, 1))

        for t in reversed(range(len(self.inputs))):
            dy = doutputs[t]  # dL/dy_t
            dWhy += np.dot(dy, self.hs[t].T)
            dby += dy

            dh = np.dot(self.Why.T, dy) + dh_next  # backprop from output and next hidden
            dh_raw = (1 - self.hs[t] ** 2) * dh  # tanh' = 1 - tanh^2

            dbh += dh_raw
            dWxh += np.dot(dh_raw, self.inputs[t].reshape(1, -1))
            dWhh += np.dot(dh_raw, self.hs[t - 1].T)

            dh_next = np.dot(self.Whh.T, dh_raw)

        # 가중치 업데이트
        for param, dparam in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
                                 [dWxh, dWhh, dWhy, dbh, dby]):
            param -= learning_rate * dparam


rnn = RNN(input_size=2, hidden_size=5, output_size=1, seq_len=3)
inputs = [np.random.randn(2) for _ in range(3)]
outputs = rnn.forward(inputs)

for t, out in outputs.items():
    print(f"Time {t} → Output: {out.ravel()}")

