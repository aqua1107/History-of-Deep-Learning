import numpy as np

class CNN:
    def __init__(self, filters, stride=1, kernel_size=(3, 3), padding=0):
        self.filters = filters
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding

    def conv2D_forward(self, batch):
        
        # batch: shape (N, C_in, H_in, W_in)
        N, C_in, H_in, W_in = batch.shape
        F, C_f, kH, kW = self.filters.shape

        assert C_in == C_f, "채널 수가 일치하지 않습니다."

        # 패딩
        if self.padding > 0:
            batch = np.pad(batch, 
                            ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 
                            mode='constant')

        # 출력 크기 계산
        H_out = (H_in + 2 * self.padding - kH) // self.stride + 1
        W_out = (W_in + 2 * self.padding - kW) // self.stride + 1

        output = np.zeros((N, F, H_out, W_out))  # 출력 텐서

        # Convolution 루프
        for n in range(N):
            for f in range(F):
                for i in range(H_out):
                    for j in range(W_out):
                        vert_start = i * self.stride
                        vert_end = vert_start + kH
                        horiz_start = j * self.stride
                        horiz_end = horiz_start + kW

                        region = batch[n, :, vert_start:vert_end, horiz_start:horiz_end]  # (C, kH, kW)
                        output[n, f, i, j] = np.sum(region * self.filters[f])  # 아다마르곱 후 합산

        return output

    def conv2D_backward(self, d_out):
        N, F, H_out, W_out = d_out.shape
        F, C_in, kH, kW = self.filters.shape
        _, _, H_in_p, W_in_p = self.X_padded.shape  # padded 입력

        dX_padded = np.zeros_like(self.X_padded)
        dW = np.zeros_like(self.filters)

        for n in range(N):
            for f in range(F):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * self.stride
                        h_end = h_start + kH
                        w_start = j * self.stride
                        w_end = w_start + kW

                        region = self.X_padded[n, :, h_start:h_end, w_start:w_end]

                        dX_padded[n, :, h_start:h_end, w_start:w_end] += d_out[n, f, i, j] * self.filters[f]
                        dW[f] += d_out[n, f, i, j] * region

        # 패딩 제거
        if self.padding > 0:
            dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_padded

        self.grad_W = dW  # 필터에 대한 gradient 저장
        return dX
    
    def relu(x):
        return np.maximum(0, x)
    
class MaxPool2D:
    
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        self.x = x # 오차역전파를 위해 저장

        N, C, H, W = x.shape
        H_out = (H - self.pool_size) // self.stride + 1
        W_out = (W - self.pool_size) // self.stride + 1

        output = np.zeros((N, C, H_out, W_out))

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        vert_start = i * self.stride
                        vert_end = vert_start + self.pool_size
                        horiz_start = j * self.stride
                        horiz_end = horiz_start + self.pool_size

                        region = x[n, c, vert_start:vert_end, horiz_start:horiz_end]
                        output[n, c, i, j] = np.max(region)

        return output

    def backward(self, grad_output):
        
        N, C, H, W = self.x.shape
        H_out = (H - self.pool_size) // self.stride + 1
        W_out = (W - self.pool_size) // self.stride + 1
        
        dx = np.zeros_like(self.x) # 미분값 초기화
        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size

                        region = self.x[n, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(region)
                        mask = (region == max_val)

                        dx[n, c, h_start:h_end, w_start:w_end] += mask * grad_output[n, c, i, j]
        return dx
    
class Flatten:
    def forward(self, x):
        self.input_shape = x.shape  # 역전파 시 사용 가능
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)
    
class Dense:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros((1, output_dim))

    def forward(self, x):
        self.x = x  # 역전파 위해 저장
        return np.dot(x, self.W) + self.b
    
    def backward(self, grad_output):
        self.grad_W = self.x.T @ grad_output
        self.grad_b = np.sum(grad_output, axis=0, keepdims=True)
        return grad_output @ self.W.T

    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 안정화
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def cross_entropy_loss(y_pred, y_true):
        # y_true는 one-hot 벡터, y_pred는 softmax된 확률값
        N = y_pred.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / N
        return loss
    
