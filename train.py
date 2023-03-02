
from models import *
from get_data import *



def main():

    '''
    This code is used to train our Linear Regression algorithm. We return the model that we will use for prediction 
    and also the X_test and y_test. This code also prints the value of theta based on the model we want to use(Normal Equation or CGS) 
    '''

    # Instanciate the LinearRegression class 
    X,y= get_data_()

    X = add_ones(X)
    X_train, X_test, y_train , y_test = split_data(X, y, 0.8)

    model= LinearRegression(2)

# Train the model
    model.fit(X_train, y_train)

# print the learned theta
    print("The value of theta obtained is :", model.theta)

    return model, X_test, y_test


if __name__=='__main__':
    print('Started!')
    main()
    print('Done')

    


