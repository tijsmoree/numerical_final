import numpy as np

m = 2
n = 2

# set constants
rho_g = 0.005
rho_w = 0.01
D = 0.01
c_hat = 4e4

# set radii
r_1 = 2
r_2 = 4
r_3 = 6
r_4 = 8
r_5 = 10

# set angles
f_1 = np.pi/8
f_2 = np.pi/8*3
f_3 = np.pi/2

# calculate amount of cells to make sure each cell has only one rho
M = m*5
N = n*4

# calculate the cell sizes
dr = r_5/M
df = f_3/N
dt = 1

# functions to calculate radii west, center and east of a cell based on j
def r_w (j):
  return dr*j
def r_c (j):
  return dr*(j+0.5)
def r_e (j):
  return dr*(j+1)

# function to determine rho using r and theta
def rho_gw (r, f):
  if r < r_1:
    return rho_g
  elif (r_2 <= r < r_3) and (f_1 <= f < f_2):
    return rho_g
  elif (r_4 <= r) and ((f < f_1) or (f >= f_2)):
    return rho_g
  else:
    return rho_w

# calculate the area of a cell using the location by using j
def area (j):
  return 0.5*df*(r_e(j)*r_e(j)-r_w(j)*r_w(j))

# the whole stiffness matrix divided by r, dr, dtheta or 
# mass matrix M inversed and multiplied with stiffness matrix S
MinvS = np.zeros((M*N, N*M))

# the whole rho vector for all the cells with either rho_white or rho_grey
rho = np.zeros(M*N)

# the initial value of c for t = 0
c0 = np.zeros(M*N)

# i walking through the cells in the theta direction
for i in range(N):
  # j walking through the cells in the r direction
  # due to horizontal ordering i*M+j is the right index for cell(i, j)
  for j in range(M):
    # northern neighbour value for MinvS
    MinvS[i*M+j, (i*M+j+M) % (N*M)] = 1/(df*r_c(j)) ** 2

    # southern neighbour value for MinvS
    MinvS[i*M+j, (i*M+j-M) % (N*M)] = 1/(df*r_c(j)) ** 2
    
    # on the east boundary or on the boundary of the circle
    if j == M-1:
      # the current cell value for MinvS
      MinvS[i*M+j, i*M+j] = -(2/(r_c(j)*r_c(j)*df*df)+r_w(j)/(r_c(j)*dr*dr))

      # the western neighbour value for MinvS
      MinvS[i*M+j, i*M+j-1] = r_w(j)/(r_c(j)*dr*dr)

    # in a cell in the middle
    else:
      # the current cell value for MinvS
      MinvS[i*M+j, i*M+j] = -(2/(r_c(j)*r_c(j)*df*df)+(r_e(j)+r_w(j))/(r_c(j)*dr*dr))

      # the eastern neighbour value for MinvS
      MinvS[i*M+j, i*M+j+1] = r_e(j)/(r_c(j)*dr*dr)

      # the western neighbour value for MinvS
      MinvS[i*M+j, i*M+j-1] = r_w(j)/(r_c(j)*dr*dr)

    # set the rho vector to rho_grey in all four grey areas and white in the rest
    rho[i*M+j] = rho_gw(j*dr, i*df)

    # calculating the c value on t = 0
    c0[i*M+j] = c_hat*np.exp(-(r_c(j)*r_c(j)))

# summing the MinvS matrix with the diagonal matrix of all rhos
# to have one matrix A for the equation c'(t) = A c(t)
A = (MinvS*D+np.diag(rho))

# set some variables
c = c0
deadcells = []
alive = True
cnt = 0
totalArea = 0.25*np.pi*r_5*r_5
B = np.eye(M*N, M*N)-dt*A

# solve the equation in time until the patient is dead with initial values c0
while alive:
  areaDead = 0
  c = np.linalg.solve(B, c)

  # calculating the dead area
  for i in range(N):
    for j in range(M):
      if c[i*M+j] > c_hat:
        areaDead += area(j)

  # determining whether the patient is dead
  if areaDead > 0.25*totalArea:
    alive = False
  
  cnt += 1

# print the day of death
print(cnt*dt)