from machine_learning_ultraupgrade import Learning as Le
import math
import json
import numpy as np
from matplotlib import pyplot as py

#フィッテングするデータをプロット
x = []
y = []
N = 30

noize = np.random.normal(
    loc=0,
    scale=1 / 100,
    size = N
)

noize = list(map(float,noize))

for i in range(N):
    x.append((2  / N) * i - 1)
    y.append(10 * math.sin(3 * x[i]) + 3 * math.cos(x[i]) + noize[i])
py.plot(x,y,label="sample_data",marker="8",linestyle='None',color="r")

with open('./temporarily_saved/learning.json') as f:
    input = json.load(f)

W_1 = input['weights'][0]
W = input['weights'][1]

f_weights = [[[Le.pow0,Le.pow1,Le.pow2,Le.pow3,Le.pow4,Le.pow5,Le.pow6,math.sin,math.cos],W_1],[[Le.pow1],W]]
z_outputs = []
for i in range(N):
    z_outputs.append(Le.neuralnetwork([x[i]],f_weights))
py.plot(x,z_outputs,label="before",marker=".",color="g")

#フィッティングさせてみる
for i in range(100):
    Le.implovement(x,y,f_weights,0.037)
#どれくらい改善されたか見てみる
with open('./temporarily_saved/learning.json') as f:
    input = json.load(f)

W_1 = input['weights'][0]
W = input['weights'][1]


f_weights = [[[Le.pow0,Le.pow1,Le.pow2,Le.pow3,Le.pow4,Le.pow5,Le.pow6,math.sin,math.cos],W_1],[[Le.pow1],W]]
z_outputs = []
for i in range(N):
    z_outputs.append(Le.neuralnetwork([x[i]],f_weights))
py.plot(x,z_outputs,label="after",marker="x",color="b")

py.legend()
py.show()
