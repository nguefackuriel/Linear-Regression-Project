
from models import *
from get_data import *



def main():

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