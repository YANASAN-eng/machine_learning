import analysis as ana
import numpy as np
import math
from matplotlib import pyplot
import json

class Learning:
    #######################################################################
    #線形変換
    #######################################################################
    @classmethod
    def linear_transformation(cls,W,z_input):
        if(type(z_input) == float):
            z_input = [z_input]
        z_output = []
        sum = 0
        for i in range(len(W)):
            for j in range(len(W[0])):
                sum = sum + W[i][j] * z_input[j]
            z_output.append(sum)
            sum = 0
        return z_output
    ########################################################################
    #非線形変換
    #########################################################################
    @classmethod
    def nonlinear_transformation(cls,fs,z_input):
        n = len(z_input)
        z_output = []
        for i in range(n):
            z_output.append(fs[i](z_input[i]))
        return z_output
    ##########################################################################
    #n層Neuralネットワーク
    ##########################################################################
    @classmethod
    def neuralnetwork(cls,z_input,f_weights):
        #f_weightsの構造
        #f_weights = [[[fs_0],W_0],[[fs_1],W_1],...,[[fs_{n - 1}],W_{n - 1}]]
        n = len(f_weights) #層の個数
        temp = z_input
        z_output = []
        for layer_number in range(n):
            z_output = cls.linear_transformation(f_weights[layer_number][1],temp)
            temp = z_output
            z_output = cls.nonlinear_transformation(f_weights[layer_number][0],temp)
            temp = z_output
        z_output = temp
        return z_output
    ##########################################################################
    #二乗誤差関数
    ##########################################################################
    @classmethod
    def error(cls,z_output,z_correct):
        sum = 0
        for i in range(len(z_output)):
            sum = sum + (z_output[i] - z_correct[i]) ** 2
        return sum / len(z_output)
    ############################################################################
    #二乗誤差関数の出力に関する微分
    ############################################################################
    @classmethod
    def dEdz_output(cls,z_output,z_correct):
        if(type(z_correct) == float):
            z_correct = [z_correct]
        dEdz_output = []
        for p in range(len(z_output)):
            dEdz_output.append(2 *  (z_output[p] - z_correct[p]) / len(z_output))
        return dEdz_output
    ############################################################################
    #\frac{\partial z_{p}^{k}}{\partial z_{q}^{k - 1}}
    @classmethod
    def dzdz(cls,z,fs,W):
        if(type(z) == float):
            z = [z]
        matrix = []
        row = []
        sum = 0
        epsilon = 0.001
        for p in range(len(W)):
            for q in range(len(W[0])):
                for j in range(len(W[0])):
                    sum = sum + W[p][j] * z[j]
                sum = W[p][q] * ana.derivative(fs[p],sum,epsilon)
                row.append(sum)
                sum = 0
            matrix.append(row)
            row = []
        return matrix
    ############################################################################
    #\frac{\partial z_{p}^{k}}{\partial W_{rq}^{k}}
    ############################################################################
    @classmethod
    def dzdW(cls,z,fs,W):
        if(type(z) == float):
            z = [z]
        tensor = []
        matrix = []
        row = []
        sum = 0
        epsilon = 0.01
        for p in range(len(W)):
            for q in range(len(W)):
                for r in range(len(W[0])):
                    if(p == q):
                        for j in range(len(W[0])):
                            sum = sum + W[p][j] * z[j]
                        sum = z[r] * ana.derivative(fs[p],sum,epsilon)
                        row.append(sum)
                        sum = 0
                    else:
                        row.append(0)
                matrix.append(row)
                row = []
            tensor.append(matrix)
            matrix = []
        return tensor
    ##########################################################################
    #matrix^{i}_{j} * vector_{i} 添字iについてはEinstein縮約をとる．
    ############################################################################
    @classmethod
    def inner_product1(cls,matrix,vector):
        u = []
        sum = 0
        for j in range(len(matrix[0])):
            for i in range(len(matrix)):
                sum = sum + matrix[i][j] * vector[i]
            u.append(sum)
            sum = 0
        return u
    ##############################################################################
    #tensor^{i}_{jk} * vector_{i} 添字iについてはEinstein縮約をとる．
    ##########################################################################
    @classmethod
    def inner_product2(cls,tensor,vector):
        matrix = []
        row = []
        sum = 0
        for p in range(len(tensor[0])):
            for q in range(len(tensor[0][0])):
                for j in range(len(tensor)):
                    sum = sum + tensor[j][p][q] * vector[j]
                row.append(sum)
                sum = 0
            matrix.append(row)
            row = []
        return matrix
    ################################################################################
    #\frac{\partial E}{\partial W_{pq}^{k}}
    #格納される順番は降順であることに注意
    ####################################################################################
    @classmethod 
    def dEdWs(cls,z,z_correct,f_weights):
        num = len(f_weights)
        z_column = []
        dEdz_output = []
        dzdzs = []
        dzdWs = []
        dEdWs = []
        temp = []
        z_column.append(z) #z_column[0]=input
        #input = z_0 to z_1 to z_2 to ... to z_n = output
        for n in range(num):
            z_column.append(cls.neuralnetwork(z_column[n],[f_weights[n]]))
        dEdz_output = cls.dEdz_output(z_column[num],z_correct)
        for n in range(num):
            dzdzs.append(cls.dzdz(z_column[n],f_weights[n][0],f_weights[n][1]))
        for n in range(num):
            dzdWs.append(cls.dzdW(z_column[n],f_weights[n][0],f_weights[n][1]))
        temp = dEdz_output
        #######################################################################
        #降順
        for n in range(num):
            if(n == 0):
                dEdWs.append(cls.inner_product2(dzdWs[(num - 1)],dEdz_output))
            else:
                temp = cls.inner_product1(dzdzs[(num - n)],temp)
                dEdWs.append(cls.inner_product2(dzdWs[(num - 1) - n],temp))
        return dEdWs
    ################################################################################
    #重みの更新(データのフィッティング)
    #############################################################################
    @classmethod
    def dEdWs_fit(cls,zs,z_corrects,f_weights):
        num = len(f_weights)
        tensor = []
        matrix = []
        row = []
        sum = 0
        for k in range(num):
            for p in range(len(f_weights[(num - 1) - k][1])):
                for q in range(len(f_weights[(num - 1) - k][1][0])):
                    for n in range(len(zs)):
                        sum = sum + cls.dEdWs(zs[n],z_corrects[n],f_weights)[k][p][q]
                    sum = sum / len(zs)
                    row.append(sum)
                    sum = 0
                matrix.append(row)
                row = []
            tensor.append(matrix)
            matrix = []
        return tensor
    ###############################################################################
    #パラメータ変更
    #################################################################################
    @classmethod
    def implovement(cls,zs,z_corrects,f_weights,param):
        dEdWs = cls.dEdWs_fit(zs,z_corrects,f_weights)
        num = len(f_weights)
        weights = []
        for k in range(num):
            weights.append(f_weights[k][1])
        for k in range(num):
            for p in range(len(weights[(num - 1) - k])):
                for q in range(len(weights[(num - 1) - k][0])):
                    weights[(num - 1) - k][p][q] = weights[(num - 1) - k][p][q] - param * dEdWs[k][p][q]
        weight_datas = {"weights":weights}
        with open('./temporarily_saved/learning.json','w') as f:
            json.dump(weight_datas,f,indent=4)
    #####################################################################################################
    #######################################################################################################
    #お試し関数
    ##########################################################################################
    def pow0(x):
        return 1
    def pow1(x):
        return x
    def pow2(x):
        return x ** 2
    def pow3(x):
        return x ** 3             
    def pow4(x):
        return x ** 4
    def pow5(x):
        return x ** 5  
    def pow6(x):
        return x ** 6   



