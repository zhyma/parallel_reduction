import csv
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# csvFile = open("data.csv","r")
# reader = csv.reader(csvFile)

mat = np.genfromtxt('reduction7_time.csv', delimiter=',')[1:,1:-1]

# for i in mat:
#     if abs(sum(i)-1) > 0.000001:
#         print(sum(i))

# print("all checked")

shape = mat.shape
x_range = ['32','64','128','256','512','1024','$2^{11}$','$2^{12}$','$2^{13}$','$2^{14}$','$2^{15}$','$2^{16}$','$2^{17}$']
y_range = ['32','64','128','256','512','1024']
x,y=np.meshgrid(range(0, shape[1]), range(0, shape[0]))
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# ax = fig.add_subplot(1, 1, 1)

mat_min = 1
mat_max = np.amax(mat)
for i in range(0,shape[0]):
    for j in range(0, shape[1]):
        if mat[i,j] < 0:
            continue
        elif mat[i,j] < mat_min:
            mat_min = mat[i,j]

lg_mat = np.zeros([shape[0],shape[1]])
for i in range(0,shape[0]):
    for j in range(0, shape[1]):
        if mat[i,j] < 0:
            lg_mat[i,j] = -1
        else:
            lg_mat[i,j] = np.log2(mat[i,j]/mat_min)

fig1, (ax)= plt.subplots(1, sharex = True, sharey = False)
ax.imshow(lg_mat, interpolation ='none', aspect = 'auto')
for (j,i),label in np.ndenumerate(mat):
    ax.text(i,j,int(label),ha='center',va='center')

ax.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)


plt.xticks(range(0, shape[1]),x_range)
plt.xlabel('number of blocks within the grid')
ax.xaxis.set_label_position('top') 
plt.yticks(range(0, shape[0]),y_range)
plt.ylabel('number of threads within the block')
plt.show() 

