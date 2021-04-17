import os
import scipy.io as sio
from sklearn.mixture import GaussianMixture
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import SparsePCA
matplotlib.use("TkAgg")


#聚类个数
cluster_nums = 7  

print(str(os.getcwd())+"\n")
for root, dirs, files in os.walk(".\\Point_Coordinate"):
    pass
print("file of the coordinates is " + str(files)+"\n")
tmp1 = sio.loadmat(("Point_Coordinate/"+files[0]))
tmp2 = list(tmp1.keys())
point_coordinates = tmp1[tmp2[-1]]

for root, dirs, files in os.walk(".\\feature_value"):
    pass
print("the feature file is:"+str(files)+"\n")
feature_nums = len(files)
print("the num of features:" + str(feature_nums)+"\n")

#提取待分析的变量
vector_order = []
R = np.zeros((3,4096))

for i in range(feature_nums):
    tmp1 = sio.loadmat(('feature_value/'+files[i]))
    tmp2 = list(tmp1.keys())
    R[i,:] = tmp1[tmp2[-1]]
    vector_order.append(tmp2[-1])
    
# 数据格式与提供方法对齐：R.shape = (n_samples,n_dimension)
R = R.T
print("the shape of the data:"+str(R.shape)+"\n")

#聚类
gm = GaussianMixture(n_components=cluster_nums, random_state=0).fit(R)
labels = gm.predict(R)

clusters = [ [] for flag in range(cluster_nums)]
clusters_order = [ [] for flag in range(cluster_nums) ]
for t in range(cluster_nums):
    for i in range(len(R)):
        if labels[i] == t:
            clusters[t].append(R[i])
            clusters_order[t].append(i)
        else:
            pass

var = {}
spca = SparsePCA(n_components=1,random_state=0)
for i in range(len(clusters)):
    spca.fit(clusters[i])
    var[i] = spca.components_
    
np.set_printoptions(precision = 10,floatmode = "fixed")
for i in range(cluster_nums):
    print(var[i])

for i in range(len(clusters)):
    print(len(clusters[i]))

#合并相同方差类，由merge_map控制
merge_map = [[1,6],[0,2,3,4,5]]
merge = []
for pair in merge_map:
    if len(pair) == 2:
        merge.append(pair)
    else:
        tmp = []
        for i in range(len(pair)-1):
            merge.append([pair[0],pair[i+1]])
print(merge)
for pair in merge:
    labels = [pair[0] if i == pair[1] else i for i in labels]

#color = {0:"#808080",3:"r",1:"b"}
#labels = list(map(lambda x:color[x],labels))



#结果可视化
for root, dirs, files in os.walk(".\\value"):
    pass
print("file of the value is " + str(files)+"\n")
tmp1 = sio.loadmat(("value/"+files[0]))
tmp2 = list(tmp1.keys())
point_value = tmp1[tmp2[-1]]

plt.figure("2D")
plt.scatter(point_coordinates,point_value,s=25,c=labels,cmap="viridis")

plt.show()
    
"""visualization"""
'''plt.figure("3D")
ax = plt.gca(projection="3d")
ax.scatter(R[:, 0], R[:, 1],R[:,1], s=20, c=labels, cmap='jet_r',alpha=0.5, marker='o')
ax.scatter(R[:, 0], R[:, 1],R[:,1], s=20, c=labels, cmap='jet_r',alpha=0.5, marker='o')

plt.figure("2D")
plt.scatter(x,u,s=50,c=labels,cmap='viridis')
plt.show()'''

"""x = np.array([[1,2,3],[1,4,3],[1,0,3],[10,2,3],[10,4,3],[10,0,3]])
gm = GaussianMixture(n_components=2, random_state=0).fit(x)

print(gm.means_)
print(gm.predict([[0,0,3],[12,3,3]]))

SPCA
X, _ = make_friedman1(n_samples=200, n_features=10, random_state=0)

#X = np.array([[1,2,3],[12,24,3],[35,46,3],[55,65,3]])
transformer = SparsePCA(n_components=10, random_state=0)
transformer.fit(X)
X_transformed = transformer.transform(X)
print(transformer.components_)
# most values in the components_ are zero (sparsity)
print(np.mean(transformer.components_ == 0))
"""


