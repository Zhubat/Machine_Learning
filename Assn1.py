
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import math

data = []

def normalise(data):
    pre_max = np.amax(data)
    pre_min = np.amin(data)

    for i in range(len(data)):
        data[i] = (data[i] - pre_min)/(pre_max - pre_min)

    return data

def data_split(in_data):
    Training_set = in_data[0:300]
    Test_set = in_data[300:]
    return Training_set, Test_set

def grad_descent(o0, o1, x1, alpha, y):
    h0 = o0 + o1*x1
    o0 = o0 + alpha*(y - h0)*1
    o1 = o1 + alpha*(y - h0)*x1
    return o0, o1, h0

def loss_function(y, h0):
    err = pow((y - h0), 2)

    return err

def RMSE(m, sum_err):
    return math.sqrt(sum_err)

def Training(data, price):
    o0 = -1
    o1 = -0.5
    lost_list = []
    err = 0
    for x in range(50):
        for i in range(len(data)):
            #for j in range(len(Training_set[i])):
            o0, o1, h0 = grad_descent(o0, o1, data[i], 0.01, price[i])
            err += loss_function(price[i], h0)
        lost_list.append(np.multiply(err, (1/len(data))))
        err = 0

    print('LOSTFUNCTION')
    print(lost_list)
    
    return o0, o1, lost_list

def Test(o0, o1, data, price):
    lost_list_test = []
    err2 = 0
    for x in range(50):
        for i in range(len(data)):
            h0 = o0*1 + o1*data[i]
            err2 += loss_function(price[i], h0)
        lost_list_test.append(np.multiply(err2, (1/len(data))))
        err2 = 0


    return lost_list_test
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
    age = data[:, 0]
    dist = data[:, 1]
    no_stores = data[:, 2]
    price = data[:, -1]
    age = normalise(age)
    dist = normalise(dist)
    no_stores = normalise(no_stores)

    age_train, age_test = data_split(age)
    dist_train, dist_test = data_split(dist)
    no_stores_train, no_stores_test = data_split(no_stores)
    Training_price, Test_price = data_split(price)

    print(age_train.shape)
    print(age_test.shape)
    print(min(age_train), max(age_train))
    #initialise theta
    

    
    o0, o1, lost_list1 = Training(age_train, Training_price)
    o0_dist, o1_dist, lost_list2 = Training(dist_train, Training_price)
    o0_no, o1_no, lost_list3 = Training(no_stores_train, Training_price)
    print(o0, o1)
    print(o0_dist, o1_dist)
    print(o0_no, o1_no)
    #print(lost_list)
    p1 = plt.plot(lost_list1)
    p2 = plt.plot(lost_list2)
    p3 = plt.plot(lost_list3)

    plt.xlabel('iteration steps'), plt.ylabel('cost function')
    plt.show()
    
    
    training_rmse = RMSE(300, np.sum(lost_list1))
    training_rmse2 = RMSE(300, np.sum(lost_list2))
    training_rmse3 = RMSE(300, np.sum(lost_list3))

    lost_list_test = Test(o0, o1, age_test, Test_price)
    lost_list_test2 = Test(o0_dist, o1_dist, dist_test, Test_price)
    lost_list_test3 = Test(o0_no, o1_no, no_stores_test, Test_price)
    test_rmse = RMSE(100, np.sum(lost_list_test))
    test_rmse2 = RMSE(100, np.sum(lost_list_test2))
    test_rmse3 = RMSE(100, np.sum(lost_list_test3))
    print('training rmse')
    print('age', training_rmse)
    print('dist', training_rmse2)
    print('stores', training_rmse3)
    print('test rmse')
    print('age', test_rmse)
    print('dist', test_rmse2)
    print('stores', test_rmse3)
    


    
