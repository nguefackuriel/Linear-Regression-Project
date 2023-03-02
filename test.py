from models import *
from get_data import *
from train import main

def test_model():

        '''
    This code is used to test our Linear Regression algorithm. We print the prediction and  the 
     This code also prints the value of theta based on the model we want to use(Normal Equation or CGS) 
    '''

    # Instanciate the LinearRegression class 
    model, X_test, y_test= main()


    # Make a prediction on X_test
    y_pred_cgs = model.predict(X_test)

    print("The predicted value is :", y_pred_cgs)

    # Compute the MSE (Evaluate both, regression and classification)
    MSE_cgs = mse(y_test, y_pred_cgs)

    print("The Mean square error is : ", MSE_cgs)


if __name__=='__main__':
    print('Started!')
    test_model()
    print('Done')