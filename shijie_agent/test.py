import numpy as np
# print(np.all([0,0,0]<[1,1,1]))
arr1 = np.array([[0,1,0],[0,0,0]])
arr2 = np.array([1,1,1])
# print(np.all(arr1<arr2))
# print(list(arr1.shape))
# shape2 = np.array([2,2])
# print(np.all(shape2<np.array(arr1.shape)))
# for i in [-1,1]:
#     for j in [-1,1]:
#         print(i,j)\
# print(arr2+[1,1,1])
arr3 = arr1
np.save("arr1.npy",arr3)
print(np.load("arr1.npy"))