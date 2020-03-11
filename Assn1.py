import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
data = []

def normalise(data):
    pre_max = np.amax(data)
    pre_min = np.amin(data)

    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = (data[i][j] - pre_min)/(pre_max - pre_min)

    return data

def data_split(in_data):
    Training_set = in_data[0:300]
    Test_set = in_data[300:]
    return Training_set, Test_set

def grad_descent(o0, o1, x1, alpha, y):
    h0 = o0 + o1*x1
    o0 = o0 + alpha*(y - h0)*x1
    o1 = o1 + alpha*(y - h0)*x1
    return o0, o1, h0

def loss_function(y, h0):
    err = pow((y - h0), 2)
    return err

if __name__ == '__main__':
    with open('house_prices.csv/house_prices.csv', newline = '') as csvfile:
        line = csv.reader(csvfile, delimiter = ' ', quotechar = '|')
        for row in line:
            if 'No' not in row[0]:
                x = ', '.join(row)
                #print(x)
                temp = x.split(',')
                #print(temp)
                data.append(temp[1:])
                

    data = np.array(data, dtype = np.float32)
    data = normalise(data)

    Training_set, Test_set = data_split(data)

    print(Training_set.shape)
    print(Test_set.shape)

    #initialise theta
    o0 = -1
    o1 = -0.5
    lost_list = []
    err = 0
    for x in range(50):
        for i in range(len(Training_set)):
            for j in range(len(Training_set[i])):
                o0, o1, h0 = grad_descent(o0, o1, Training_set[i][0], 0.01, Training_set[i][-1])
                err += loss_function(Training_set[i][-1], h0)
        lost_list.append(err)
        err = 0
    
    print(o0, o1)
    #print(lost_list)
    plt.plot(lost_list)
    plt.ylabel('iteration steps'), plt.xlabel('cost function')
    plt.show()