# Import librairies

import numpy as np
from sklearn.utils import shuffle
import get_data

# cgs

def cgs(A):
  """
    Q,R = cgs(A)
    Apply classical Gram-Schmidt to mxn rectangular/square matrix. 

    Parameters
    -------
    A: mxn rectangular/square matrix   

    Returns
    -------
    Q: mxn square matrix
    R: nxn upper triangular matrix

  """
  # ADD YOUR CODES
  m=A.shape[0] # get the number of rows of A
  n=A.shape[1] # get the number of columns of A

  R= np.zeros((n,n)) # create a zero matrix of nxn
  Q= np.ones((m,n)) # copy A (deep copy)

  
  for k in range(n):
    w = A[:,k]
    for j in range(k-1):
      R[j,k] = Q[:,j] @ w
    for j in range(k-1):
      w = w - R[j,k] * Q[:,j]
    R[k,k] = np.linalg.norm(w, 2)
    Q[:,k] = w/R[k,k]
    
    

  return Q, R


# Implement BACK SUBS
def backsubs(U, b):

  """
  x = backsubs(U, b)
  Apply back substitution for the square upper triangular system Ux=b. 

  Parameters
  -------
    U: nxn square upper triangular array
    b: n array
    

  Returns
  -------
    x: n array
  """

  n= U.shape[1]
  x= np.zeros((n,))
  b_copy= np.copy(b)

  if U[n-1,n-1]==0.0:
    if b[n-1] != 0.0:
      print("System has no solution.")
  
  else:
    x[n-1]= b_copy[n-1]/U[n-1,n-1]
  for i in range(n-2,-1,-1):
    if U[i,i]==0.0:
      if b[i]!= 0.0:
        print("System has no solution.")
    else:
      for j in range(i,n):
        b_copy[i] -=U[i,j]*x[j]
      x[i]= b_copy[i]/U[i,i]
  return x

# Add ones
def add_ones(X):
  ones_ = np.ones((X.shape[0], 1))

  # ADD YOUR CODES
  return np.hstack((ones_, X))

#X,y= get_data()

#X = add_ones(X)


def split_data(X,Y, train_size):
  # ADD YOUR CODES
  # shuffle the data before splitting it
  train_index = int(float(train_size) * len(X))
  X, Y = shuffle(X,Y)

  
  X_train, X_test = X[:train_index], X[train_index:]
  y_train, y_test = Y[:train_index], Y[train_index:]
  return X_train, X_test, y_train , y_test

#X_train, X_test, y_train , y_test = split_data(X, y, 0.8)

def mse(y, y_pred):
    # ADD YOUR CODES
    return np.mean((y-y_pred)**2)


def normalEquation(X,y):

    # ADD YOUR CODES
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)), X.T), y)


class LinearRegression:

  def __init__(self, arg):
      # ADD YOUR CODES
      
      self.arg = input('Which method do you want to use? 1:Normal equation or 2:CGS, choose a number ')
      
  def fit(self,x,y):
      # ADD YOUR CODES
    self.x = x
    self.y = y
    if self.arg == '1':
      self.theta = normalEquation(x, y)
    elif self.arg == '2':
      Q, R = cgs(x)
      self.theta=backsubs(R, np.dot(Q.T, y))

      


    
  def predict(self,x):
      #ADD YOUR CODES
      self.x = x
      return np.dot(self.x, self.theta)