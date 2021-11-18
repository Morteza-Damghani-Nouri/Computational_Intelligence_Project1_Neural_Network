import time

import numpy as np
import random
import pickle
import random
# generate random Gaussian values
from numpy.random import seed
from numpy.random import randn
import math
import matplotlib.pyplot as plt


# This function updates weights and biases
def matrix_updater(input_matrix, input_grad, input_eta, input_raw, input_column, input_batch_size):
    i = 0
    while i < input_raw:
        j = 0
        while j < input_column:
            input_matrix[i][j] = input_matrix[i][j] - input_eta * input_grad / input_batch_size
            j += 1
        i += 1


# This function calculates the subtraction of two matrix and returns the output matrix
def matrix_subtraction(first_matrix, second_matrix):
    if len(first_matrix) != len(second_matrix):
        print("There is an error in matrix_summation because of the sizes of two matrix")
        return 0
    output_matrix = []
    i = 0
    while i < len(first_matrix):
        output_matrix.append(first_matrix[i] - second_matrix[i])
        i += 1
    return output_matrix


# This function multiplies the input_number to the input_matrix and returns the result matrix
def matrix_multiplication_by_number(input_number, input_matrix, input_raw, input_column):
    output_matrix = []
    i = 0
    while i < input_raw:
        temp_matrix = []
        j = 0
        while j < input_column:
            temp_matrix.append(input_number * input_matrix[i][j])
            j += 1
        output_matrix.append(temp_matrix)
        i += 1
    return output_matrix


# This function calculates the multiplication of two input matrix
def matrix_multiplication_by_matrix(first_matrix, second_matrix):
    if isinstance(second_matrix[0], int) or isinstance(second_matrix[0], np.float64) or isinstance(second_matrix[0], float):
        second_matrix = [second_matrix]
    if isinstance(first_matrix[0], int) or isinstance(first_matrix[0], np.float64) or isinstance(first_matrix[0], float):
        first_matrix = [first_matrix]
    first_matrix_columns_count = len(first_matrix[0])
    second_matrix_columns_count = len(second_matrix[0])
    output_matrix = []
    i = 0
    while i < len(first_matrix):
        k = 0
        temp_matrix = []
        while k < second_matrix_columns_count:
            j = 0
            temp_result = 0
            while j < first_matrix_columns_count:
                temp_result += first_matrix[i][j] * second_matrix[j][k]
                j += 1
            temp_matrix.append(temp_result)
            k += 1
        output_matrix.append(temp_matrix)
        i += 1
    return output_matrix



# This function generates the batches according to size of batch
def batch_generator(input_batch_size, input_data):
    output_list = []
    main_counter = 0
    number_of_batches = math.ceil(len(input_data) / input_batch_size)
    while main_counter < number_of_batches:
        i = main_counter * batch_size
        temp_list = []
        while i < (main_counter + 1) * batch_size:
            temp_list.append(input_data[i])
            i += 1
        output_list.append(temp_list)
        main_counter += 1
    return output_list


# This function generates a zero matrix with the given dimensions
def zero_matrix_generator(input_raw, input_column):
    output_matrix = []
    i = 0
    while i < input_raw:
        j = 0
        temp_list = []
        while j < input_column:
            temp_list.append(0)
            j += 1
        output_matrix.append(temp_list)
        i += 1
    return output_matrix


# This function returns the element number of the maximum
def maximum_element_number_finder(input_list):
    maximum_element = 0
    maximum_amount = input_list[0]
    i = 1
    while i < len(input_list):
        if input_list[i] > maximum_amount:
            maximum_element = i
            maximum_amount = input_list[i]
        i += 1
    return maximum_element


# This function receives an input array and calculates the sigmoid prime of the input
def sigmoid_prime(input_array):
    output_matrix = []
    i = 0
    while i < len(input_array):
        output_matrix.append(sigmoid(input_array[i]) * (1 - sigmoid(input_array[i])))
        i += 1
    return output_matrix


def my_plotter(input_error_list):
    print(len(input_error_list))
    summation = 0
    i = 0
    y = []
    while i < len(input_error_list):
        if i == 200 or i == 400 or i == 600 or i == 800:
            y.append(summation / 200)
            summation = 0

        summation += input_error_list[i]

        i += 1
    y.append(summation / 200)
    # x axis values
    x = [1, 2, 3, 4, 5]
    # plotting the points
    plt.plot(x, y)
    # naming the x axis
    plt.xlabel("Epoch")
    # naming the y axis
    plt.ylabel("Average Error")
    # function to show the plot
    plt.show()
    v = 0
    while v < len(y):
        print("The average error in " + str(v + 1) + " epoch is: " + str(y[v]))
        v += 1


# This function receives an input and returns the sigmoid amount of the input
def sigmoid(input_number):
    return 1 / (1 + math.exp(-input_number))


# loading training set features
f = open("Datasets/train_set_features.pkl", "rb")
train_set_features2 = pickle.load(f)
f.close()

# reducing feature vector length
features_STDs = np.std(a=train_set_features2, axis=0)
train_set_features = train_set_features2[:, features_STDs > 52.3]

# changing the range of data between 0 and 1
train_set_features = np.divide(train_set_features, train_set_features.max())

# loading training set labels
f = open("Datasets/train_set_labels.pkl", "rb")
train_set_labels = pickle.load(f)
f.close()

# ------------
# loading test set features
f = open("Datasets/test_set_features.pkl", "rb")
test_set_features2 = pickle.load(f)
f.close()

# reducing feature vector length
features_STDs = np.std(a=test_set_features2, axis=0)
test_set_features = test_set_features2[:, features_STDs > 48]

# changing the range of data between 0 and 1
test_set_features = np.divide(test_set_features, test_set_features.max())

# loading test set labels
f = open("Datasets/test_set_labels.pkl", "rb")
test_set_labels = pickle.load(f)
f.close()

# ------------
# preparing our training and test sets - joining datasets and lables
train_set = []
test_set = []

for i in range(len(train_set_features)):
    label = np.array([0, 0, 0, 0])
    label[int(train_set_labels[i])] = 1
    label = label.reshape(4, 1)
    train_set.append((train_set_features[i].reshape(102, 1), label))

for i in range(len(test_set_features)):
    label = np.array([0, 0, 0, 0])
    label[int(test_set_labels[i])] = 1
    label = label.reshape(4, 1)
    test_set.append((test_set_features[i].reshape(102, 1), label))

# shuffle
random.shuffle(train_set)
random.shuffle(test_set)

# print size
# print(len(train_set))  # 1962
# print(len(test_set))  # 662

start = time.time()
# Selecting 200 random data from training dataset
random_training_data = []
random_training_elements = []   # This list is used to stop choosing repetitive elements of training_set
i = 1
while i <= 200:
    while True:
        random_number = random.randint(0, 1961)
        if random_number not in random_training_elements:
            random_training_elements.append(random_number)
            random_training_data.append(train_set[random_number])
            break
    i += 1


# Generating random weight matrix
# seed random number generator
seed(1)
# generate some Gaussian values
gaussian_random_values = randn(102*150)

first_weights_array = []
i = 0
while i <= 101:
    j = i * 150
    raw = []
    while j < (i + 1) * 150:
        raw.append(gaussian_random_values[j])
        j += 1
    first_weights_array.append(raw)
    i += 1


# seed random number generator
seed(1)
# generate some Gaussian values
gaussian_random_values = randn(150*60)

second_weights_array = []
i = 0
while i <= 149:
    j = i * 60
    raw = []
    while j < (i + 1) * 60:
        raw.append(gaussian_random_values[j])
        j += 1
    second_weights_array.append(raw)
    i += 1


# seed random number generator
seed(1)
# generate some Gaussian values
gaussian_random_values = randn(60*4)

third_weights_array = []
i = 0
while i <= 59:
    j = i * 4
    raw = []
    while j < (i + 1) * 4:
        raw.append(gaussian_random_values[j])
        j += 1
    third_weights_array.append(raw)
    i += 1



errors_list = []
# generating biases
first_biases_array = zero_matrix_generator(1, 102)
second_biases_array = zero_matrix_generator(1, 150)
third_biases_array = zero_matrix_generator(1, 60)
# Output is calculated here
correct_result_counter = 0
epoch = 1
batch_size = 10
eta = 1
batch_list = batch_generator(batch_size, random_training_data)

while epoch <= 5:
    i = 0
    while i < len(batch_list):
        current_batch = batch_list[i]
        grad_w_first_layer = 0
        grad_b_first_layer = 0
        grad_w_second_layer = 0
        grad_b_second_layer = 0
        grad_w_third_layer = 0
        grad_b_third_layer = 0
        for m in current_batch:
            training_data_features = m[0]
            # First Part
            first_result_list = []
            first_z = []
            j = 0
            while j < 150:
                k = 0
                first_result = 0
                while k < 102:
                    first_result += training_data_features[k] * first_weights_array[k][j] + first_biases_array[0][k]
                    k += 1
                first_result_list.append(sigmoid(first_result))
                first_z.append(first_result)
                j += 1
            # Second Part
            second_result_list = []
            second_z = []
            j = 0
            while j < 60:
                k = 0
                second_result = 0
                while k < 150:
                    second_result += first_result_list[k] * second_weights_array[k][j] + second_biases_array[0][k]
                    k += 1
                second_result_list.append(sigmoid(second_result))
                second_z.append(second_result)
                j += 1

            # Last Part
            final_result_list = []
            final_z = []
            j = 0
            while j < 4:
                k = 0
                final_result = 0
                while k < 60:
                    final_result += second_result_list[k] * third_weights_array[k][j] + third_biases_array[0][k]
                    k += 1
                final_result_list.append(sigmoid(final_result))
                final_z.append(final_result)
                j += 1

            h = 0
            error_summation = 0
            while h < 4:
                error_summation += math.pow(final_result_list[h] - m[1][h], 2)
                h += 1
            errors_list.append(error_summation)


            # Computing gradient descents for last layer
            # This part is for weight derivation
            for p in second_result_list:
                grad_w_third_layer += matrix_multiplication_by_number(2 * p, matrix_multiplication_by_matrix(matrix_subtraction(final_result_list, m[1]), np.array([sigmoid_prime(final_z)]).transpose()), 1, 1)[0][0]
            # This part is for bias derivation
            grad_b_third_layer += matrix_multiplication_by_number(2, matrix_multiplication_by_matrix(matrix_subtraction(final_result_list, m[1]), np.array([sigmoid_prime(final_z)]).transpose()), 1, 1)[0][0]

            cost_ak_rond = []
            for o in third_weights_array:
                summation = 0
                for l in o:
                    summation += matrix_multiplication_by_number(2 * l, matrix_multiplication_by_matrix(matrix_subtraction(final_result_list, m[1]), np.array([sigmoid_prime(final_z)]).transpose()), 1, 1)[0][0]
                cost_ak_rond.append(summation)


            # Computing gradient descents for third layer
            # This part is for weight derivation
            for p in first_result_list:
                grad_w_second_layer += matrix_multiplication_by_number(p, matrix_multiplication_by_matrix(cost_ak_rond, np.array([sigmoid_prime(second_z)]).transpose()), 1, 1)[0][0]
            # This part is for bias derivation
            grad_b_second_layer += matrix_multiplication_by_matrix(cost_ak_rond, np.array([sigmoid_prime(second_z)]).transpose())[0][0]

            cost_am_rond = []
            for o in second_weights_array:
                summation = 0
                for l in o:
                    summation += matrix_multiplication_by_number(l, matrix_multiplication_by_matrix(cost_ak_rond, np.array([sigmoid_prime(second_z)]).transpose()), 1, 1)[0][0]
                cost_am_rond.append(summation)


            # Computing gradient descents for third layer
            # This part is for weight derivation
            for p in training_data_features:
                grad_w_first_layer += matrix_multiplication_by_number(p, matrix_multiplication_by_matrix(cost_am_rond, np.array([sigmoid_prime(first_z)]).transpose()), 1, 1)[0][0]
            # This part is for bias derivation
            grad_b_first_layer += matrix_multiplication_by_matrix(cost_am_rond, np.array([sigmoid_prime(first_z)]).transpose())[0][0]


        # Updating weights
        matrix_updater(first_weights_array, grad_w_first_layer, eta, 102, 150, batch_size)
        matrix_updater(second_weights_array, grad_w_second_layer, eta, 150, 60, batch_size)
        matrix_updater(third_weights_array, grad_w_third_layer, eta, 60, 4, batch_size)
        # Updating biases
        matrix_updater(first_biases_array, grad_b_first_layer, eta, 1, 102, batch_size)
        matrix_updater(second_biases_array, grad_b_second_layer, eta, 1, 150, batch_size)
        matrix_updater(third_biases_array, grad_b_third_layer, eta, 1, 60, batch_size)

        i += 1





    print("epoch " + str(epoch) + " finished")





    epoch += 1
stop = time.time()
my_plotter(errors_list)
print("The plot is ready")
print("The executation time is: " + str((stop - start) / 60) + " minutes")





































