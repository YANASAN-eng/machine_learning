import numpy as np
import math
from complex import Complex as Comp


class Linear:
    matrix = []
    def __init__(self,matrix):
        self.matrix.append(matrix)   
    #行列の転置
    @classmethod
    def transpose(cls,A):
        m = len(A)
        n = len(A[0])
        A_transpose = np.empty((n,m),dtype=Comp)
        for i in range(m):
            for j in range(n):
                A_transpose[j][i] = A[i][j]
        return A_transpose
    #複素共役
    @classmethod 
    def conjugate(cls,A):
        m = len(A)
        n = len(A[0])
        A_conjugate = np.empty((n,m),dtype=Comp)
        for i in range(m):
            for j in range(n):
                A_conjugate[j][i] = Comp.conjugate(A[i][j])
        return A_conjugate
    #行列の和
    @classmethod
    def add(cls,A,B):
        if(len(A) == len(B) and len(A[0]) == len(B[0])):
            C = np.empty((len(A),len(A[0])),dtype=Comp)
            for i in range(0,len(A)):
                for j in range(0,len(A[0])):
                    C[i][j] = Comp.add(A[i][j],B[i][j])
            return C
        else:
            return "入力された二つの行列の型が異なるため，和を定義することができません．"
    #マイナス
    @classmethod
    def minus(cls,A):
        C = np.empty((len(A),len(A[0])),dtype=Comp)
        for i in range(len(A)):
            for j in range(len(A[0])):
                C[i][j] = Comp.minus(A[i][j])
        return C
    #スカラー倍
    @classmethod
    def scalar(cls,a,A):
        C = np.empty((len(A),len(A[0])),dtype=Comp)
        for i in range(len(A)):
            for j in range(len(A[0])):
                C[i][j] = Comp.times(a,A[i][j])
        return C
    #行列の掛け算
    @classmethod
    def times(cls,A,B):
        if(len(A[0]) == len(B)):
            C = np.empty((len(A),len(B[0])),dtype=Comp)
            for i in range(len(A)):
                for j in range(len(B[0])):
                    C[i][j] = Comp(0,0)
            for i in range(len(A)):
                for k in range(len(B[0])):
                    for j in range(len(A[0])):
                        C[i][k] = Comp.add(C[i][k],Comp.times(A[i][j],B[j][k]))
            return C
        else:
            return "第一引数の行列の列数と第二引数の行列の行数が異なるため行列の積を定義する事ができません．"
    #UL分解
    @classmethod
    def LU(cls,A):
        if(len(A) == len(A[0])):
            L = np.empty((len(A),len(A)),dtype=Comp)
            U = np.empty((len(A),len(A)),dtype=Comp)
            for i in range(len(A)):
                for j in range(len(A[0])):
                    L[i][j] = Comp(0,0)
                    U[i][j] = Comp(0,0)
            LU = []
            for i in range(len(A)):
                for j in range(len(A)):
                    if(i == j):
                        L[i][j] = Comp(1,0)
            for i in range(len(A)):
                for j in range(len(A)):
                    if(i == 0):
                        U[i][j] = A[i][j]
                    else:
                        if(i > j):
                            L[i][j] = A[i][j]
                            for k in range(j):
                                L[i][j] = Comp.add(L[i][j],Comp.minus(Comp.times(L[i][k],U[k][j])))
                            L[i][j] = Comp.times(L[i][j],Comp.reciprocal(U[j][j]))
                        if(i <= j):
                            U[i][j] = A[i][j]
                            for k in range(i):
                                U[i][j] = Comp.add(U[i][j],Comp.minus(Comp.times(L[i][k],U[k][j])))        
            LU.append(L)
            LU.append(U)
            return LU             
        else:
            return "正方行列ではないのでLU分解できません"
    #行列式を算出
    @classmethod
    def determination(cls,A):
        U = cls.LU(A)[1]
        det = Comp(1,0)
        for i in range(len(U)):
            if(U[i][i] == Comp(0,0)):
                return Comp(0,0)
            else:
                det = Comp.times(det,U[i][i])
        return det
    #逆行列計算
    @classmethod
    def inverse(cls,A):
        L = cls.LU(A)[0]
        U = cls.LU(A)[1]
        E = np.empty((len(A),len(A)),dtype=Comp)
        for i in range(len(A)):
            for j in range(len(A)):
                if(i == j):
                    E[i][j] = Comp(1,0)
                else:
                    E[i][j] = Comp(0,0)
        L_inverse = np.empty((len(A),len(A)),dtype=Comp)
        for i in range(len(A)):
            for j in range(len(A)):
                if(i == j):
                    L_inverse[i][j] = Comp(1,0)
                else:
                    L_inverse[i][j] = Comp(0,0)
        U_inverse = np.empty((len(A),len(A)),dtype=Comp)
        for i in range(len(A)):
            for j in range(len(A)):
                if(i == j):
                    U_inverse[i][j] = Comp(1,0)
                else:
                    U_inverse[i][j] = Comp(0,0)
        s = Comp(0,0)
        for i in range(len(A)):
            for j in range(len(A[0])):
                if(i == 0):
                    continue
                else:
                    s = Comp(0,0)
                    for k in range(i):
                        s = Comp.add(s,Comp.times(L[i][k],L_inverse[k][j]))
                    L_inverse[i][j] = Comp.add(L_inverse[i][j],Comp.minus(s))
        for i in range(len(A)):
            for j in range(len(A)):
                if(i == 0):
                    if(j < len(A) - 1):
                        continue
                    else:
                        U_inverse[len(A) - 1][j] = Comp.reciprocal(U[len(A) - 1][len(A) - 1])
                else:
                    s = Comp(0,0)
                    for k in range(i):
                        s = Comp.add(s,Comp.times(U[(len(A) - 1) - i][(len(A) - 1) -k],U_inverse[(len(A) - 1) -k][j]))
                    U_inverse[(len(A) - 1) - i][j] = Comp.times(Comp.add(E[(len(A) - 1) - i][j],Comp.minus(s)),Comp.reciprocal(U[(len(A) - 1) - i][(len(A) - 1) - i]))
        return cls.times(U_inverse,L_inverse)
    #対称行列の場合のCholesky分解
    @classmethod
    def cholesky(cls,A):
        L = cls.LU(A)[0]
        L_transpose = cls.transpose(cls.LU(A)[0])
        L_transpose_inverse = cls.inverse(L_transpose)
        U = cls.LU(A)[1]
        D = cls.times(U,L_transpose_inverse)
        cholesky = []
        cholesky.append(L)
        cholesky.append(D)
        cholesky.append(L_transpose)
        return cholesky
    #出力
    @classmethod 
    def output(cls,str):
        print(str)
    @classmethod
    def output(cls,A):
        for i in range(len(A)):
            for j in range(len(A[0])):
                if(j < len(A[0]) - 1):
                    Comp.output(A[i][j])
                elif(j == len(A[0]) - 1):
                    Comp.output(A[i][j])
                    print(" ")
    #固有値分解
    #ノルム
    @classmethod
    def norm(cls,A):
        sum = 0
        for i in range(len(A)):
            for j in range(len(A[0])):
                sum = sum + A[i][j].real * A[i][j].real + A[i][j].image * A[i][j].image
        return  math.sqrt(sum)
    #QR分解
    #単位行列として初期化
    @classmethod
    def E(cls,A):
        m = len(A)
        n = len(A[0])
        for i in range(m):
            for j in range(n):
                if(i == j):
                    A[i][j] = Comp(1,0)
                else:
                    A[i][j] = Comp(0,0)
        return A
    #行列の直和
    @classmethod
    def direction(cls,A,B):
        C = np.empty((len(A) + len(B),len(A[0]) + len(B[0])),dtype=Comp)
        for i in range(len(A) + len(B)):
            for j in range(len(A[0]) + len(B[0])):
                if(i < len(A) and j < len(A[0])):
                    C[i][j] = A[i][j]
                elif(len(A) <= i and len(A[0]) <= j):
                    C[i][j] = B[i - len(A)][j - len(A[0])]
                else:
                    C[i][j] = Comp(0,0)
        return C
    #QR分解
    #http://marupeke296.com/oxmath_no3_eigenvalues.html
    @classmethod
    def QR(cls,A):
        n = len(A)
        R = A
        Q = np.empty((n,n),dtype=Comp)
        Q = cls.E(Q)
        H = np.empty((n,n),dtype=Comp)
        QR = []
        cls.E(H)
        for count in range(n):
            x = np.empty((n - count,1),dtype=Comp)
            y = np.empty((n - count,1),dtype=Comp)
            u = np.empty((n - count,1),dtype=Comp)
            H_temp = np.empty((n - count,n -count),dtype=Comp)
            Delta = np.empty((count,count),dtype=Comp)
            E = np.empty((n - count,n - count),dtype=Comp)
            E = cls.E(E)
            if(count == 0):
                for i in range(n - count):
                    x[i][0] = R[count + i][count]
                for i in range(n - count):
                    if(i == 0):
                        y[i][0] = Comp.times(Comp.argument(x[0][0]),Comp(cls.norm(x),0))
                    else:
                        y[i][0] = Comp(0,0)
                    u[i][0] = Comp.add(x[i][0],Comp.minus(y[i][0]))
                if(cls.norm(u) == 0):
                    H = E
                    R = cls.times(H,R)
                    Q = cls.times(Q,cls.inverse(H))
                else:
                    H_temp = cls.add(E,cls.scalar(Comp.times(Comp(-2,0),Comp.reciprocal(Comp.times(Comp(cls.norm(u),0),Comp(cls.norm(u),0)))),cls.times(u,cls.conjugate(u))))
                    H = H_temp
                    R = cls.times(H,R)
                    Q = cls.times(Q,cls.inverse(H))
            elif(count < n - 1):
                Delta = cls.E(Delta)
                for i in range(n - count):
                    x[i][0] = R[count + i][count]
                for i in range(n - count):
                    if(i == 0):
                        y[i][0] = Comp.times(Comp.argument(x[0][0]),Comp(cls.norm(x),0))
                    else:
                        y[i][0] = Comp(0,0)
                    u[i][0] = Comp.add(x[i][0],Comp.minus(y[i][0]))
                if(cls.norm(u) == 0):
                    H = cls.direction(Delta,E)
                    R = cls.times(H,R)
                    Q = cls.times(Q,cls.inverse(H))
                else:
                    H_temp = cls.add(E,cls.scalar(Comp.times(Comp(-2,0),Comp.reciprocal(Comp.times(Comp(cls.norm(u),0),Comp(cls.norm(u),0)))),cls.times(u,cls.conjugate(u))))
                    H = cls.direction(Delta,H_temp)
                    R = cls.times(H,R)
                    Q = cls.times(Q,cls.inverse(H))
        QR.append(Q)
        QR.append(R)
        return QR
    @classmethod
    def eigenvalue(cls,A):
        B = A
        eigenvalue = []
        N = 30
        for i in range(N):
            B = cls.times(cls.QR(B)[1],cls.QR(B)[0])
        for i in range(len(A)):
            eigenvalue.append(B[i][i])
            Comp.output((eigenvalue[i]))
            print('')
    #Jordan標準形をもとめる
    #https://www.jstage.jst.go.jp/article/iscie1988/15/7/15_7_320/_pdf
