import numpy as np
import numpy.linalg as lin
# from scipy.sparse import diags
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import csv


m = 6
n = 6

rho_g = 0.005
rho_w = 0.01
D = 0.01
c_hat = 4e4

r_1 = 2
r_2 = 4
r_3 = 6
r_4 = 8
r_5 = 10

th_1 = np.pi / 8
th_2 = np.pi / 8 * 3
th_3 = np.pi / 2

M = m * 5
N = n * 4

dr = r_5 / M
df = th_3 / N
dt = 1

# functions to calculate radii west, center and east of a cell based on j
def r_w (j):
    return dr * j
def r_c (j):
    return dr * (j + 0.5)
def r_e (j):
    return dr * (j + 1)

# function to determine rho using r and theta
def rho_gw (r, th):
    if r < r_1:
        return rho_g
    elif (r_2 <= r < r_3) and (th_1 <= th < th_2):
        return rho_g
    elif (r_4 <= r) and ((th < th_1) or (th >= th_2)):
        return rho_g
    else:
        return rho_w

# the whole stiffness matrix divided by r, dr, dtheta or mass matrix M inversed and multiplied with stiffness matrix S
MinvS = np.zeros((M * N, N * M))

# the whole rho vector for all the cells with either rho_white or rho_grey
rho = np.zeros(M * N)

# the initial value of c for t = 0
c0 = np.zeros(M * N)

# i walking through the cells in the theta direction
for i in range(N):
    # j walking through the cells in the r direction
    # due to horizontal ordering i * M + j is the right index for cell(i, j)
    for j in range(M):
        # northern neighbour value for MinvS
        MinvS[i * M + j, (i * M + j + M) % (N * M)] = 1 / (df * r_c(j)) ** 2

        # southern neighbour value for MinvS
        MinvS[i * M + j, (i * M + j - M) % (N * M)] = 1 / (df * r_c(j)) ** 2
        
        # on the left boundary or in the middle of the circle
        if j == 0:
            # the current cell value for MinvS
            MinvS[i * M + j, i * M + j] =  -(2 / (r_c(j) * r_c(j) * df * df) + r_e(j) / (r_c(j) * dr * dr))

            # the eastern neighbour value for MinvS
            MinvS[i * M + j, i * M + 1] = r_e(j) / (r_c(j) * dr * dr)

        # on the right boundary or on the boundary of the circle
        elif j == M - 1:
            # the current cell value for MinvS
            MinvS[i * M + j, i * M + j] =  -(2 / (r_c(j) * r_c(j) * df * df) + r_w(j) / (r_c(j) * dr * dr))

            # the western neighbour value for MinvS
            MinvS[i * M + j, i * M + j - 1] = r_w(j) / (r_c(j) * dr * dr)

        # in a cell in the middle
        else:
            # the current cell value for MinvS
            MinvS[i * M + j, i * M + j] =  -(2 / (r_c(j) * r_c(j) * df * df) + (r_e(j) + r_w(j)) / (r_c(j) * dr * dr))

            # the eastern neighbour value for MinvS
            MinvS[i * M + j, i * M + j + 1] = r_e(j) / (r_c(j) * dr * dr)

            # the western neighbour value for MinvS
            MinvS[i * M + j, i * M + j - 1] = r_w(j) / (r_c(j) * dr * dr)

        # set the rho vector to rho_grey in all four grey areas and white in the rest
        rho[i * M + j] = rho_gw((j) * dr, (i) * df)

        # calculating the c value on t = 0
        c0[i * M + j] = c_hat * np.exp(-(r_c(j) * r_c(j)))


# summing the MinvS matrix with the diagonal matrix of all rhos to have one matrix A for the equation c'(t) = A c(t)
A = (MinvS * D + np.diag(rho))
print("Calculating the matrix is done!")

c = c0
deadcells = []
alive = True
cnt = 0
while alive:    
    c += lin.solve(A, c) * dt    
    deadcells.append(sum(i > chat for i in c))
    print(max(c))    
    if sum(i > chat for i in c) / (M * N) > 0.25:
        alive = False
    cnt += 1

t = cnt * dt
