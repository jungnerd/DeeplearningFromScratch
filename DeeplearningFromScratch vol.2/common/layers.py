import numpy as np
# import common.functions import *
from common.util import im2col, col2im


"""
계층의 '구현 규칙'
- 모든 계층은 `forward()`와 `backward()` 메서드를 가진다.
- 모든 계층은 인스턴스 변수인 `params`와 `grads`를 가진다.
"""

class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None
        
    def forward(self, x):
        W, = self.params # tuple 인 self.params를 벗겨서  W에 받기 위해 W, 를 사용한다
        out = np.matmul(x, W)
        self.x = x
        return out
    
    def backward(self, dout):
        W, = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        self.grads[0][...] = dW # self.grads[0] = dW 을 하게 되면 grads를 수정할 때 dW 값도 바뀌게 된다. 이를 막기 위해 self.grads[0][...]을 사용한다
        return dx
    
    
class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None
        
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    
    def backward(self, dout):
        dx = dout * (1.0 -self.out) * self.out
        return dx
    
    
class Affine: # TODO : MatMul class를 이용해서 구현하기
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None
        
    def forward(self, x):
        W, b = self.params
        out = np.matmul(x, W) + b
        self.x = x
        return out
    
    def backward(self, dout):
        W, b = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        db = np.sum(dout, axis=0)
        
        self.grads[0][...] = dW
        self.grads[1][...] = db
        
        return dx
    