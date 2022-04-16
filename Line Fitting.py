# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 15:05:52 2022

@author: sharma
"""

import csv
import numpy as np
from numpy.linalg import eig, norm
import matplotlib.pyplot as plt
from scipy.linalg import svd

filename = "linear_regression_dataset.csv"

# initilise the age and insurance charges lists
age = []
charges = []

# Calculate the covariance values for the matrix
def calculateCovariance(x, y):
    x_mean = sum(x) / float(len(x))
    y_mean = sum(y) / float(len(y))

    
    sub_x = [elem - x_mean for elem in x]
    sub_y = [elem - y_mean for elem in y]

    summ = np.matmul(sub_x, sub_y)
    N = float(len(age)-1)
    
    cov = summ/N
    return cov

# Create the Covariance matrix
def covariance(array):
    c = [[calculateCovariance(a,b) for a in array] for b in array]
    return c

def leastSquare(a, c):
    # values of x and y for Least Square fitting
    x = np.array(a)
    y = np.array(c)
    
    # Coefficient estimates
    n = np.size(x)
    mean_x = np.mean(x) 
    mean_y = np.mean(y)
    
    sxy = np.sum(y * x) - n * mean_x * mean_y
    sxx = np.sum(x * x) - n * mean_x * mean_x
    
    b1 = sxy / sxx
    b0 = mean_y - b1 * mean_x
    b = (b0, b1)
    
    print("Estimated coefficients:\nb_0 = {} \ \nb_1 = {}".format(b[0], b[1]))  # b = {}")
    return x, y, b


def totalLeastSquare(tx, ty):    
    tx = np.array(tx)
    if len(tx.shape) == 1:
        n = 1
        tx = tx.reshape(len(tx),1)
    else:
        n = np.array(tx).shape[1] # the number of variable of X
    
    Z = np.vstack((tx.T,ty)).T
    U, s, Vt = svd(Z, full_matrices=True)

    V = Vt.T
    Vxy = V[:n, n:]
    Vyy = V[n:, n:]
    a_tls = - Vxy  / Vyy # total least squares soln
    
    Xtyt = - Z.dot(V[:,n:]).dot(V[:,n:].T)
    Xt = Xtyt[:,:n] # X error
    y_tls = (tx+Xt).dot(a_tls)

    fro_norm = norm(tx, 'fro')#Frobenius norm
    
    return y_tls, tx + Xt, a_tls, fro_norm

# reading csv file
with open(filename, 'r') as csvfile:

    # creating a csv reader object
    csvreader = csv.reader(csvfile)
    
    # extracting fiekd names thrugh first row
    title = next(csvreader)
    print("test", title)
    
    # extracting each data row one by one
    for row in csvreader:
        if row[0].isdigit() == True:
            age.append(float(row[0]))
            charges.append(float(row[6]))
    
    c1 = np.cov(age)
    c2 = np.cov(charges)
    mat = np.stack((age, charges), axis = 0 )
    #print("zzz", mat.shape)
    c3 = np.cov(mat)
    print(c3)
    
    print("Total number of rows: ", csvreader.line_num)
    
    covariance_matrix = np.array(covariance(mat))
    print("Covariance Matrix = \n", covariance_matrix)
    #print(covariance_matrix.shape)
    
    #v1 = calculate_covariance(age, age)
    #print(v1)
    #v12 = calculate_covariance(age, charges)
    #print(v12)
    #v2 = calculate_covariance(charges, charges)
    #print(v2)
    
    eigenvalue,eigenvector= eig(covariance_matrix)
    print("Eigenvalues of the Covariance Matrix are = \n", eigenvalue)
    print("Eigenvectors of the Covariance Matrix are = \n", eigenvector)
    
    #eigenvalue,eigenvector= eig(np.cov(mat))
    #print("test1", eigenvalue)
    #print("test2", eigenvector)
    
    slope= (eigenvector[1][1]-eigenvector[1][0])/(eigenvector[0][1]-eigenvector[0][0])
    #y2-y1/x2-x2         
    x_v1, y_v1 = eigenvector[:, 0]

    x_v2, y_v2 = eigenvector[:, 1]

    
    m1 = y_v1 / x_v1
    m2 = y_v2 / x_v2
    
    x_vec = np.linspace(min(age), max(age))
    y1 = m1 * x_vec
    y2 = m2 * x_vec
    origin = [0, 0]   
 
    plt.title("Health Insurance costs based on the person’s age")
    plt.xlabel("Age of the person")
    plt.ylabel("Health Insurnace charges")
    plt.scatter(age, charges)
    plt.plot(x_vec, y1, color = 'red', label = "E1")
    plt.plot(x_vec, y2, color = 'green', label = "E2")        
    #plt.show()
    

    plt.quiver(origin, [x_v1, y_v1], color='orange', label = "Vector 1", scale=21)
    plt.quiver(origin, [x_v2, y_v2], color='y', label = "Vector 1", scale=21)
    plt.rcParams["figure.figsize"] = (8,5)
    plt.legend(loc = 'upper left')
    plt.savefig("covariance plot.png", bbox_inches='tight')
    plt.show()
    
    age_x, charges_y , const = leastSquare(age, charges)
    plt.scatter(age_x, charges_y, s = 30, color = 'm', marker = "x", label = "Dataset")
    y_pred = const[0] + const[1] * age_x
    plt.plot(age_x, y_pred, color = 'black', label = "Least Square")
    #plt.xlabel("Age of the person")
    #plt.ylabel("Health Insurnace charges")
    #plt.title("Least Square Fitting for the dataset")
    #plt.legend(loc = 'upper left')
    #plt.savefig("least square.png", bbox_inches='tight')
    #plt.show()
    
    tls_chargesy, tls_agex, tls_soln, fnorm = totalLeastSquare(age, charges)
    #plt.scatter(age, charges, s = 30, color = 'm', marker = "x", label = "Dataset")
    #y_estimate = np.array()
    plt.plot(tls_agex, tls_chargesy, color = 'blue', label = "Total Least Square")
    plt.xlabel("Age of the person")
    plt.ylabel("Health Insurnace charges")
    plt.title("LS and TLS Fitting for the dataset")
    plt.legend(loc = 'upper left')
    plt.xlim((0, max(age) + 5))
    plt.ylim((0, max(charges)))
    #plt.savefig("total least square.png", bbox_inches='tight')
    plt.savefig("Line fitting with LS and TLS.png", bbox_inches='tight')
    plt.show()
    
    length = len(mat[0])
    age=[1 for a in mat if len(a)!=length]
    
    if(sum(age)>0):
        raise Exception("length of vectors not same")
    

    
