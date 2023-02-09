# linear_regression.py
# Song Li
# 11/7/2021

import pandas as pd
import math

def LinearRegression(src_data):
    N = len(src_data)
    
    # variables holding the coordinates
    x = src_data['x']
    y = src_data['y']

    mean_x = x.mean()
    mean_y = y.mean()

    cov = ((x - mean_x) * (y - mean_y)).sum() / (N - 1) # covariance
    std_dev_x = math.sqrt(((x - mean_x) ** 2).sum() / (N - 1)) # standard devariation
    squared_sdx = std_dev_x ** 2 # squared standard devariation

    beta = cov / squared_sdx
    alpha = mean_y - beta * mean_x
    line = 'y = {}x + {}'.format(round(beta, 2), round(alpha, 2))

    return beta, alpha, line, round(mean_x, 2), round(mean_y, 2), round(cov, 2), round(std_dev_x, 2), round(squared_sdx, 2), round(beta, 2), round(alpha, 2)

def Prediction(beta, alpha, ind_var):
    dep_var = ind_var * beta + alpha
    return dep_var

if __name__ == '__main__':
    src_data = pd.read_csv('linear_regression_source_file.csv')
    
    beta, alpha, line, mean_x, mean_y, cov, std_dev_x, squared_sdx, beta, alpha = LinearRegression(src_data)
    print(f"Linear regression equation: {line}")
    print(f"mean_x: {mean_x}")
    print(f"mean_y: {mean_y}")
    print(f"cov: {cov}")
    print(f"std_dev_x: {std_dev_x}")
    print(f"squared_sdx: {squared_sdx}")
    print(f"beta: {beta}")
    print(f"alpha: {alpha}")

    ind_var = input("Enter an independent value: ")
    dep_var = Prediction(round(beta, 2), round(alpha, 2), int(ind_var))
    print(f"The predicted value is: {dep_var}")