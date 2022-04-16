# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 19:48:55 2022

@author: sharm
"""

# Singular-value decomposition
import numpy as np
from numpy import array
from numpy.linalg import eig


# Calculating the the single value decomposition matirices 
# and the homography matrix for a given input matrix

# Input:
#   A: Given Input matrix  

# Output:
#   U: Orthogonal matrix formed using eigen vectors of AAt.
#   S: Diagonal matrix Sigma.
#   VT: Transpose of the second orthogonal matrix formed using eigen vectors of AtA.
#   H: Homography matrix of the given input matrix.

# define a matrix
#point projection to frame 1
x1, y1= 5, 5
x2, y2= 150, 5
x3, y3= 150, 150
x4, y4= 5, 150

#point projection to frame 2
xp1, yp1= 100, 100
xp2, yp2= 200, 80
xp3, yp3= 220, 80
xp4, yp4= 100, 200

#A matrix
A = np.array([[-x1, -y1, -1, 0, 0, 0, x1*xp1, y1*yp1, xp1], 
              [0, 0, 0, -x1, -y1, -1, x1*yp1, y1*yp1, yp1],
              [-x2, -y2, -1, 0, 0, 0, x2*xp2, y2*yp2, xp2], 
              [0, 0, 0, -x2, -y2, -1, x2*yp2, y2*yp2, yp2],
              [-x3, -y3, -1, 0, 0, 0, x3*xp3, y3*yp3, xp3], 
              [0, 0, 0, -x3, -y3, -1, x3*yp3, y3*yp3, yp3],
              [-x4, -y4, -1, 0, 0, 0, x4*xp4, y4*yp4, xp4], 
              [0, 0, 0, -x4, -y4, -1, x4*yp4, y4*yp4, yp4]])


print("\nInput Matrix A: \n",A,"\n")

# calculate eigenvectors and eigenvalues after operating on matrix A
AT = np.transpose(A)
AT_A = np.matmul(AT, A)  # np.dot(AT, A)
A_AT = np.matmul(A, AT)

eigenvalue_AAt_unsorted, eigenvector_AAt = eig(A_AT)
eigenvalue_AtA, eigenvector_AtA = eig(AT_A)

#sort the eigenvalue 
eigenvalue_AAt = eigenvalue_AAt_unsorted[eigenvalue_AAt_unsorted.argsort()[::-1]]
#print("test", eigenvalue_AAt)
U = eigenvector_AAt[:,eigenvalue_AAt_unsorted.argsort()[::-1]]
VT = np.real(eigenvector_AtA[:, eigenvalue_AtA.argsort()[::-1]])
print("U matrix: \n", U,"\n")
print("VT matrix:\n", VT,"\n")

#extracting singular values and constructing a diagonal matrix
sing = np.sqrt(np.abs(eigenvalue_AAt_unsorted))
sigma= np.diag(sing)
print("Sigma matrix: \n", sigma,"\n")
    
#print("Eigen Values of AtA \n",eigenvalue_AtA,"\n")
#finding the index of the least eigen value in AtA
least_eigenvalue = np.where(np.abs(eigenvalue_AtA) == np.amin(np.abs(eigenvalue_AtA)))[0]

# Homography matrix
H = VT[:,least_eigenvalue]
H = np.reshape(VT[:, least_eigenvalue],(3,3))
print("Homography Matrix \n", H,"\n")