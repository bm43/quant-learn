import numpy as np

class Convolution:
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
        oh = int((h + 2 * ph - kh) / sh + 1)
        ow = int((w + 2 * pw - kw) / sw + 1)
        output = np.zeros((n, self.out_channels, oh, ow))
        
        x = np.pad(x, [(0, 0), (0, 0), (ph, ph), (pw, pw)], mode='constant')
        for i in range(n):
            for j in range(self.out_channels):
                for k in range(c):
                    for m in range(oh):
                        for n in range(ow):
                            r = m*sh
                            c = n*sw
                            output[i, j, m, n] += (x[i, k, r:r+kh, c:c+kw] * self.weights[j, k]).sum()
                output[i, j] += self.bias[j]
                
        return output