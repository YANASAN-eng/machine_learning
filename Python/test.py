import json
from complex import Complex as Comp
from linear_upgrade import Linear as Lin
import numpy as np

#jsonファイル読み込み
with open('./temporarily_saved/linear.json') as f:
    mat = json.load(f)
m = len(mat)
n = len(mat[0])

#配列を取得
A = np.empty((m,n),dtype=Comp)
A = Lin.E(A)
for i in range(m):
    for j in range(n):
        if('+i' in mat[i][j]):
            if(mat[i][j].split('+i')[1] == ''):
                A[i][j].real = float(mat[i][j].split("+i")[0])
                A[i][j].image = 1
            else:
                A[i][j].real = float(mat[i][j].split("+i")[0])
                A[i][j].image = float(mat[i][j].split("+i")[1])
        elif('-i' in mat[i][j]):
            if(mat[i][j].split("-i")[0] == ''):
                if(mat[i][j].split('-i')[1] == ''):
                    A[i][j].real = 0
                    A[i][j].image = -1
                else:
                    A[i][j].real = 0
                    A[i][j].image = -float(mat[i][j].split("-i")[1])
            else:
                if(mat[i][j].split('-i')[1] == ''):
                    A[i][j].real = float(mat[i][j].split("-i")[0])
                    A[i][j].image = -1
                else:
                    A[i][j].real = float(mat[i][j].split("-i")[0])
                    A[i][j].image = -float(mat[i][j].split("-i")[1])
        elif(str(mat[i][j])[0] == 'i'):
            if(mat[i][j].split('i')[1] == ''):
                A[i][j].real = 0
                A[i][j].image = 1
            else:
                A[i][j].real = 0
                A[i][j].image = float(mat[i][j].split('i')[1])
        else:
            A[i][j].real = float(mat[i][j])
            A[i][j].image = 0
        
#入力された行列
print('your inputting matrix')
Lin.output(A)
print('---------------------------------------------------------')
#LU分解
print('the matrix is LU decomposition of your inputting matrix.')
print('L')
Lin.output(Lin.LU(A)[0])
print('---------------------------------------------------------')
print('U')
Lin.output(Lin.LU(A)[1])
print('---------------------------------------------------------')
print('the matrix is verification about LU composition of your inputting matrix')
Lin.output(Lin.times(Lin.LU(A)[0],Lin.LU(A)[1]))
print('---------------------------------------------------------')
#逆行列
print('the matrix is inverse matrix for your inputting matrix')
Lin.output(Lin.inverse(A))
print('---------------------------------------------------------')
#行列式
print('the value is determination of your inputting matrix.')
Comp.output(Lin.determination(A))
print('')
print('---------------------------------------------------------')
#QR分解
print('the matrix is QR decomposition of your inputting matrix.')
print('Q')
Lin.output(Lin.QR(A)[0])
print('---------------------------------------------------------')
print('R')
Lin.output(Lin.QR(A)[1])
print('---------------------------------------------------------')
print('the matrix is verification about QR composition of your inputting matrix')
Lin.output(Lin.times(Lin.QR(A)[0],Lin.QR(A)[1]))
print('---------------------------------------------------------')
#固有値
print('the values are eigenvalues of your inputting matrix.')
Lin.eigenvalue(A)
