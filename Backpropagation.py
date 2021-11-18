import numpy as np
import random
import pickle
import random
# generate random Gaussian values
from numpy.random import seed
from numpy.random import randn
import math



# This function calculates the summation of two matrix and returns the output matrix
def matrix_summation(first_matrix, second_matrix):
    if len(first_matrix) != len(second_matrix):
        print("There is an error in matrix_summation because of the sizes of two matrix")
        return 0
    output_matrix = []
    i = 0
    while i < len(first_matrix):
        output_matrix.append(first_matrix[i] + second_matrix[i])
        i += 1
    return output_matrix


# This function multiplies the input_number to the input_matrix and returns the result matrix
def matrix_multiplication_by_number(input_number, input_matrix):
    output_matrix = []
    i = 0
    while i < len(input_matrix):
        output_matrix.append(input_matrix[i] * input_number)
        i += 1
    return output_matrix


# This function calculates the multiplication of two input matrix
def matrix_multiplication_by_matrix(first_matrix, second_matrix):
    second_matrix_columns_count = len(second_matrix[0])
    first_matrix_columns_count = len(first_matrix[0])
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


# This function generates bias array
def bias_generator(input_number):
    output = []
    i = 0
    while i < input_number:
        output.append(0)
        i += 1
    return output


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




# generating biases
first_biases_array = bias_generator(102)
second_biases_array = bias_generator(150)
third_biases_array = bias_generator(60)
# Output is calculated here
correct_result_counter = 0
first_z = []
second_z = []
final_z = []
epoch = 0
batch_size = 10
batch_list = batch_generator(batch_size, random_training_data)
print("The length of batch_list is: " + str(len(batch_list)))

while epoch <= 4:
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
            j = 0
            while j < 150:
                k = 0
                first_result = 0
                while k < 102:
                    first_result += training_data_features[k] * first_weights_array[k][j] + first_biases_array[k]
                    k += 1
                first_result_list.append(sigmoid(first_result))
                first_z.append(first_result)
                j += 1
            # Second Part
            second_result_list = []
            j = 0
            while j < 60:
                k = 0
                second_result = 0
                while k < 150:
                    second_result += first_result_list[k] * second_weights_array[k][j] + second_biases_array[k]
                    k += 1
                second_result_list.append(sigmoid(second_result))
                second_z.append(second_result)
                j += 1

            # Last Part
            final_result_list = []
            j = 0
            while j < 4:
                k = 0
                final_result = 0
                while k < 60:
                    final_result += second_result_list[k] * third_weights_array[k][j] + third_biases_array[k]
                    k += 1
                final_result_list.append(sigmoid(final_result))
                final_z.append(final_result)
                j += 1

            # Computing gradient descents for each layer
            # This part is for weight derivation
            for p in second_result_list:
                grad_w_third_layer = matrix_summation(grad_w_third_layer, matrix_multiplication_by_number(2 * p, matrix_multiplication_by_matrix(matrix_subtraction(final_result_list, m[1]), sigmoid_prime(final_z))))





            # This part is for bias derivation

















        i += 1






    epoch += 1






























i = 0
while i < 200:
    training_data_features = random_training_data[i][0]
    first_result_list = []
    j = 0
    while j < 150:
        k = 0
        first_result = 0
        while k < 102:
            first_result += training_data_features[k] * first_weights_array[k][j]
            k += 1
        first_result_list.append(sigmoid(first_result))
        first_z.append(first_result)
        j += 1




    second_result_list = []
    j = 0
    while j < 60:
        k = 0
        second_result = 0
        while k < 150:
            second_result += first_result_list[k] * second_weights_array[k][j]
            k += 1
        second_result_list.append(sigmoid(second_result))
        second_z.append(second_result)
        j += 1


    final_result_list = []
    j = 0
    while j < 4:
        k = 0
        final_result = 0
        while k < 60:
            final_result += second_result_list[k] * third_weights_array[k][j]
            k += 1
        final_result_list.append(sigmoid(final_result))
        final_z.append(final_result)
        j += 1


    maximum_element_number = maximum_element_number_finder(final_result_list)
    if random_training_data[i][1][maximum_element_number] == 1:
        correct_result_counter += 1

    i += 1

i = 1

print("The accuracy is: " + str(correct_result_counter / 2) + "%")





print(len(first_z))
print(len(second_z))
print(len(final_z))




while i <= 5:
# Updating third weights array









    i += 1
































