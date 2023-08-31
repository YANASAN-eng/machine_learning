from complex import Complex as Comp
from linear_upgrade import Linear as Li
import numpy as np

#微分法
def derivative(f,x,h):
    return (f(x + h) - f(x)) / h
#積分法
def integral(f,a,b,n):
    sum = 0
    delta = (b - a) / n
    for i in range(n):
        sum = sum + f(a + delta * i) * delta
    return sum
