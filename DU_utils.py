import numpy as np
# import matplotlib.pyplot as plt
import random
import pandas


class Module:
    def __init__(self, name):
        self.name = name
        self.train_mode = False
        
    def __call__(self, *args):
        return self.forward(*args)
    
    def train(self):
        self.train_mode = True
        for m in self.modules():
            m.train()
        
    def eval(self):
        self.train_mode = False
        for m in self.modules():
            m.eval()
        
    def modules(self):
        ms = []
        for attr in self.__dict__:
            m = self.__dict__[attr]
            if isinstance(m, Module):
                ms.append(m)
        return ms
    
    def params(self):
        ps = []
        for attr in self.__dict__:
            p = self.__dict__[attr]
            if isinstance(p, Parameter):
                ps.append(p)
            
        ms = self.modules()
        for m in ms:
            ps.extend(m.params())
        return ps
    
    def info(self, n):
        ms = self.modules()
        output = f"{self.name}\n"
        for m in ms:
            output += ('  '*(n+1)) + f"{m.info(n+1)}\n"
        return output[:-1]
    
    def __repr__(self):
        return self.info(0)
    
class Initializer:
    def __init__(self, name):
        self.name = name
        
    def __call__(self, *args):
        return self.apply(*args)
        
class GaussInitializer(Initializer):
    # where :math:`\mu` is the mean and :math:`\sigma` the standard
    # deviation. The square of the standard deviation, :math:`\sigma^2`,
    # is called the variance.
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        
    def apply(self, value):
        value[...] = np.random.normal(self.mu, self.sigma, value.shape)
    
class Parameter:
    def __init__(self, value):
        self.value = value
        self.delta = np.zeros(value.shape)
        
    def zero_grad(self):
        self.delta[...] = 0
        
class Linear(Module):
    def __init__(self, input_feature, output_feature):
        super().__init__("Linear")
        self.input_feature = input_feature
        self.output_feature = output_feature
        self.weights = Parameter(np.zeros((input_feature, output_feature)))
        self.bias = Parameter(np.zeros((1, output_feature)))
        
        # 权重初始化 
        initer = GaussInitializer(0, np.sqrt(2 / input_feature))  # np.sqrt(2 / input_feature)
        initer.apply(self.weights.value)
        
    def forward(self, x):
        self.x_save = x.copy()
        return x @ self.weights.value + self.bias.value
    
    #AB = C  G
    #dB = A.T @ G
    #dA = G @ B.T
    def backward(self, G):
        self.weights.delta += self.x_save.T @ G
        self.bias.delta += np.sum(G, 0)  #值复制
        return G @ self.weights.value.T
    
class ReLU(Module):
    def __init__(self, inplace=True):
        super().__init__("ReLU")
        self.inplace = inplace
        
    # 亿点点
    def forward(self, x):
        self.negative_position = x < 0
        if not self.inplace:
            x = x.copy()
            
        x[self.negative_position] = 0
        return x
    
    def backward(self, G):
        if not self.inplace:
            G = G.copy()
            
        G[self.negative_position] = 0
        return G
    
class PReLU(Module):
    def __init__(self, num_feature, inplace=False):
        super().__init__("PReLU")
        self.inplace = inplace
        self.coeff = Parameter(np.zeros((num_feature)))
        
    def forward(self, x):
        if not self.inplace:
            x = x.copy()
            
        for channel in range(x.shape[1]):
            view = x[:, channel]
            negative_position = view < 0
            view[negative_position] *= self.coeff.value[channel]
        return x
    
class Softmax(Module):
    def __init__(self):
        super().__init__("Softmax")
        
    def forward(self, x):
        expx = np.exp(x)
        sumx = np.sum(expx, axis=1, keepdims=True)
        return expx / sumx

class SWish(Module):
    def __init__(self):
        super().__init__("SWish")
        
    def forward(self, x):
        self.x_save = x.copy()
        self.sx = sigmoid(x)
        return x * self.sx
    
    def backward(self, G):
        return G * (self.sx + self.x_save * self.sx * (1 - self.sx))
    
class Dropout(Module):
    def __init__(self, prob_keep=0.5, inplace=True):
        super().__init__("Dropout")
        self.prob_keep = prob_keep
        self.inplace = inplace
        
    def forward(self, x):
        if not self.train_mode:
            return x
        
        self.mask = np.random.binomial(size=x.shape, p=1 - self.prob_keep, n=1)
        if not self.inplace:
            x = x.copy()
            
        x[self.mask] = 0
        x *= 1 / self.prob_keep  # rescale
        return x
    
    def backward(self, G):
        if not self.inplace:
            G = G.copy()
        G[self.mask] = 0
        G *= 1 / self.prob_keep
        return G
    
    
class Flatten(Module):
    def __init__(self):
        super().__init__("Flatten")
        
    def forward(self, x):
        self.in_shape = x.shape
        out = x.reshape(self.in_shape[0], -1)
        return out
    
    def backward(self, G):
        return G.reshape(self.in_shape)
    
class ModuleList(Module):
    def __init__(self, *args):
        super().__init__("ModuleList")
        self.ms = list(args)
        
    def modules(self):
        return self.ms
    
    def forward(self, x):
        for m in self.ms:
            x = m(x)
        return x
    
    def backward(self, G):
        for i in range(len(self.ms)-1, -1, -1):
            G = self.ms[i].backward(G)
        return G

class MaxPooling2d(Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__("MaxPooling2d")
        self.kernel_size = kernel_size
        self.stride = stride
        
    def forward(self, x):
        ib, ic, ih, iw = x.shape
        oh, ow = int(np.ceil((ih - self.kernel_size) / self.stride) + 1), int(np.ceil((iw - self.kernel_size) / self.stride) + 1)
        output = np.zeros((ib, ic, oh, ow))
        for b in range(ib):
            for c in range(ic):
                for oy in range(oh):
                    for ox in range(ow):
                        ix = ox * self.stride
                        iy = oy * self.stride
                        output[b, c, oy, ox] = x[b, c, iy, ix]
                        
                        for kx in range(self.kernel_size):
                            for ky in range(self.kernel_size):
                                ix = ox * self.stride + kx
                                iy = oy * self.stride + ky
                                if ix < iw and iy < ih:
                                    output[b, c, oy, ox] = max(x[b, c, iy, ix], output[b, c, oy, ox])
        return output
    
class BatchNormalization(Module):
    def __init__(self, in_feature, momentum=0.9, eps=1e-8):
        self.mu = 0
        self.var = 1
        self.momentum = momentum
        self.eps = eps
        self.in_feature = in_feature
        self.gamma = Parameter(np.ones(in_feature))
        self.beta = Parameter(np.zeros(in_feature))
        
    def forward(self, x):
        
        if not self.train_mode:
            y = (x - self.mu) / np.sqrt(self.var + self.eps)
            return y * self.gamma.value.reshape(1, -1, 1, 1) + self.beta.value.reshape(1, -1, 1, 1)
        
        self.b_mu = np.mean(x, axis=(0, 2, 3), keepdims=True)
        self.b_var = np.var(x, axis=(0, 2, 3), keepdims=True)
        self.y = (x - self.b_mu) / np.sqrt(self.b_var + self.eps)
        self.mu = self.b_mu * self.momentum + self.mu * (1 - self.momentum)
        
        n = x.size / x.shape[1]
        unbiased_var = self.b_var * n / (n - 1)
        self.var = unbiased_var * self.momentum + self.var * (1 - self.momentum)
        return self.y * self.gamma.value.reshape(1, -1, 1, 1) + self.beta.value.reshape(1, -1, 1, 1)
    
    def backward(self, G):
        self.gamma.delta = np.sum(G * self.y, axis=(0, 2, 3))
        self.beta.delta = np.sum(G, axis=(0, 2, 3))
        return G * self.gamma.value.reshape(1, -1, 1, 1) / np.sqrt(self.b_var + self.eps)




class Conv2d(Module):
    def __init__(self,in_feature, out_feature, kernel_size, padding = 0, stride = 1):
        super().__init__("Conv2d")
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

#         self.kernel = Parameter(np.zeros((out_feature, in_feature, kernel_size, kernel_size)))# 就是这样定义的
        self.kernel = Parameter(
                    np.array([
                            [0,0,0],
                            [1,1,0],
                            [0,0,0] 
                    ])[None][None])
        self.bias = Parameter(np.zeros((out_feature)))# 每一组kernel 配一个bias
        initer = GaussInitializer(0,2/np.sqrt(in_feature))
#         initer.apply(self.kernel.value)

        
    def forward(self,x):
        # the input :img and kernel
        self.in_shape = x.shape
        ib,ic,ih,iw = self.in_shape
        self.khalf = self.kernel_size//2
        # output
        self.oh = (ih-self.kernel_size + 2*self.padding)//self.stride + 1
        self.ow = (iw-self.kernel_size + 2*self.padding)//self.stride + 1
        self.output = np.zeros((ib,self.out_feature,self.oh,self.ow))
    
        # column
        self.column = np.zeros((ib,self.kernel_size*self.kernel_size*ic ,self.oh*self.ow))
        # k_col
        self.k_col = self.kernel.value.reshape((self.out_feature,-1))

        
        for b in range(ib):
            for channel in range(ic):
                for oy in range(self.oh):# oy ox 指的是输出在输出图像的坐标【跟v1 v2的cy cx不一样】
                    for ox in range(self.ow):
                        for ky in range(self.kernel_size):
                            for kx in range(self.kernel_size):
                            # where the pixel value goes in column
                                column_y = self.kernel_size**2*channel + ky*self.kernel_size + kx
                                column_x = oy*self.ow + ox # ow的格数大小就是kernel横向取了几次
                                # where the pixel value comes from img
                                iy = oy*self.stride+ky - self.padding
                                ix = ox*self.stride+kx - self.padding

                                # 如果iy ix超出边界(可能进入了padding地带)，就不处理
                                if iy >=0 and iy < ih and ix >= 0 and ix < iw:
                                    self.column[b,column_y, column_x] = x[b,channel,iy,ix]
            
            self.output[b] = (self.k_col @ self.column[b]).reshape(-1,self.oh,self.ow) + self.bias.value.reshape((self.out_feature))       
        return self.output  
        
    def backward(self,G):# G : G_in : dL/d output(this layer)
        ib,ic,ih,iw = self.in_shape # the shape of x  [input of the current layer]
        
        # 1.update part
        # k_col @ column = output
        for b in range(ib):
            # 首先三维的G[b] 肯定是要reshape成2维。因为G[b]：d output(this layer)，所以shape与output[b]是一样的
            # output[b]是[out_feature,oh,ow]
            self.kernel.delta += (G[b].reshape(-1,self.oh*self.ow)@self.column[b].T).reshape(self.kernel.value.shape) # column[b].T shape: (oh*ow,kh*kw*channel)
        
        self.bias.delta += np.sum(G,axis = (0,2,3)) # 因为G的第一个通道是out_feature,对应的就是有多少组kernel
        
        # 2.pass back part
        self.Gout = np.zeros((self.in_shape))

        for b in range(ib):
            # dcolumn我们这里仅仅作为当前图片的dcolumn
            dcolumn = self.k_col.T @ G[b].reshape(self.out_feature,-1) # k_col.T shape: (kw*kh*ic,out_feature)
            # dcolumn 和column shape是一样的
  
            for channel in range(ic):
                for oy in range(self.oh):# oy ox 指的是输出在输出图像的坐标【跟v1 v2的cy cx不一样】
                    for ox in range(self.ow):
                        for ky in range(self.kernel_size):
                            for kx in range(self.kernel_size):
                            # where the pixel value comes from column
                                column_y = self.kernel_size**2*channel + ky*self.kernel_size + kx
                                column_x = oy*self.ow + ox # ow的格数大小就是kernel横向取了几次
                                # where the pixel value goes to img 可参考 notability 笔记 “输入输出坐标的推导”
                                iy = oy*self.stride+ky - self.padding
                                ix = ox*self.stride+kx - self.padding

                                # 如果iy ix超出边界(可能进入了padding地带)，就不处理
                                if iy >=0 and iy < ih and ix >= 0 and ix < iw:
                                    self.Gout[b,channel,iy,ix] += dcolumn[column_y, column_x]
                                    #上面之所以使用+= 是因为在im2col的时候，一个img像素会搬到column的多个地方
                                    #（由于是滑动窗口会重叠），也就是说一个像素会在column不同地方出现，所以回传的时候
                                    #有多个地方贡献梯度
        
        return self.Gout

if __name__ == "__main__":   
    itensor = np.arange(25).reshape(5,5)[None][None]
    conv2d = Conv2d(in_feature=1,out_feature=1,kernel_size=3,padding = 0)
    # forward
    print("DU's forward")
    print(conv2d(itensor))

    # backward
    print("DU's backward")
    G = np.array([[[[11., 13., 15.],
         [21., 23., 25.],
         [31., 33., 35.]]]])
    print(conv2d.backward(G))
