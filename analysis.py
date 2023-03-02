import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from get_data import get_data_


X, y = get_data_()

print(len(X)), print(len(y))
plt.subplot(2,1,1)
plt.plot(X)
plt.subplot(2,1,2)
plt.plot(y)
plt.show()