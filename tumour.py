import numpy as np
import numpy.linalg as lin
from scipy.sparse import diags
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n = 2
m = 2

rhog = 0.005
rhow = 0.01
D = 0.01
chat = 4e4

r_upper = 10

M = m * 5
N = n * 4

dr = r_upper / M
df = np.pi/2 / N
dt = 0.001

S = np.zeros((M*N,N*M))

for i in range(N):
    for j in range(M):
        S[i*M+j,(i*M+j+M)%(N*M)] = dr/(df*dr*(j+0.5))
        S[i*M+j,(i*M+j-M)%(N*M)] = dr/(df*dr*(j+0.5))
        if j == 0:
            S[i*M+j,i*M+j] = -(2*dr/(df*dr*(j+0.5))+(j+1)*df)/(dr*df*dr*(j+0.5))
            S[i*M+j,i*M+1] = (j+1)/(dr*dr*(j+0.5))
        elif j == M-1:
            S[i*M+j,i*M+j] = -(2*dr/(df*dr*(j+0.5))+j*df)/(dr*df*dr*(j+0.5))
            S[i*M+j,i*M+j-1] = (j)/(dr*dr*(j+0.5))
        else:
            S[i*M+j,i*M+j] = -(2*dr/(df*dr*(j+0.5))+(2*j+1)*df)/(dr*df*dr*(j+0.5))
            S[i*M+j,i*M+j+1] = (j+1)/(dr*dr*(j+0.5))
            S[i*M+j,i*M+j-1] = (j)/(dr*dr*(j+0.5))

vec = np.ones(M*N)

for i  in range(N):
    for j in range(M):
        if j < n:
            vec[i*M+j] = rhog
        elif (2*n <= j < 3*n) and (m <= i < 3*m):
            vec[i*M+j] = rhog
        elif (4*n <= j) and ((i < m) or (i >= 3*m)):
            vec[i*M+j] = rhog
        else:
            vec[i*M+j] = rhow

A = (S*D+np.diag(vec))

vec = np.ones(M*N)

for i  in range(N):
    for j in range(M):
        vec[i*M+j] = chat*np.exp(-1*((j+0.5)*dr)**2)

eig = lin.eigvals(A)
      
c = vec
deadcells = []
print("matrix done!")
ALIVE = True
cnt = 0
while ALIVE:    
    c += lin.solve(A,c)*dt    
    deadcells.append(sum(i > chat for i in c))
    print(max(c))    
    if sum(i > chat for i in c)/(M*N) > 0.25:
        ALIVE = False
    cnt+=1

t = cnt*dt
