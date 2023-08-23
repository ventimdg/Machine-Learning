import numpy as np

#setup

array = [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]]
vector1 = [1,2,3,4,5]
vector2 = [6,7,8,9,10]

nparray = np.array(array)
npvector1 = np.array(vector1)
npvector2 = np.array(vector2)

#6.1.1

einsum_trace = np.einsum('ii', nparray)
np_trace = np.trace(nparray)

print("einsum trace minus np trace \n")
print(einsum_trace - np_trace)
print("\n")


#6.1.2
matrix_vector_einsum = np.einsum('ij,j->i', nparray, npvector1)
matrix_vector_numpy = nparray @ npvector1

diff1 = matrix_vector_einsum - matrix_vector_numpy

print("Matrix Vector Multiplication: Norm of einsum answer minus regular matrix operation answer\n")
print(np.linalg.norm(diff1))
print("\n")


#6.1.3
einsum_outer_product = np.einsum('i,j->ij', npvector1, npvector2)
regular_outer_product = np.outer(npvector1, npvector2)

diff2 = einsum_outer_product - regular_outer_product

print("Outer Product: Norm of einsum answer minus np.outer answer\n")
print(np.linalg.norm(diff2))
print("\n\n")