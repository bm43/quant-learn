import numpy as np

class Deconvolution:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.bias = np.zeros(out_channels)

    def forward(self, x):
        n, c, h, w = x.shape
        kh, kw = self.kernel_size, self.kernel_size
        sh, sw = self.stride, self.stride
        ph, pw = self.padding, self.padding
        oh = (h - 1) * sh - 2 * ph + kh
        ow = (w - 1) * sw - 2 * pw + kw
        output = np.zeros((n, self.out_channels, oh, ow))

        for i in range(n):
            for j in range(self.out_channels):
                for k in range(c):
                    output[i, j] += np.kron(x[i, k], self.weights[j, k])
                output[i, j] = output[i, j][ph:ph+oh, pw:pw+ow] + self.bias[j]
            
        return output