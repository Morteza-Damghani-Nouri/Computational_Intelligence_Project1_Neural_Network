import numpy as np
import random
import pickle
import random
# generate random Gaussian values
from numpy.random import seed
from numpy.random import randn
import math


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
for i in range(102):
    j = i * 150
    raw = []
    while j < (i + 1) * 150:
        raw.append(gaussian_random_values[j])
        j += 1
    first_weights_array.append(raw)



# seed random number generator
seed(1)
# generate some Gaussian values
gaussian_random_values = randn(150*60)

second_weights_array = []
i = 0
for i in range(150):
    j = i * 60
    raw = []
    while j < (i + 1) * 60:
        raw.append(gaussian_random_values[j])
        j += 1
    second_weights_array.append(raw)



# seed random number generator
seed(1)
# generate some Gaussian values
gaussian_random_values = randn(60*4)

third_weights_array = []
i = 0
for i in range(60):
    j = i * 4
    raw = []
    while j < (i + 1) * 4:
        raw.append(gaussian_random_values[j])
        j += 1
    third_weights_array.append(raw)







# Output is calculated here
correct_result_counter = 0

i = 0
for i in range(200):
    training_data_features = random_training_data[i][0]
    first_result_list = []
    j = 0
    for j in range(150):
        k = 0
        first_result = 0
        while k < 102:
            first_result += training_data_features[k] * first_weights_array[k][j]
            k += 1
        first_result_list.append(sigmoid(first_result))





    second_result_list = []
    j = 0
    for j in range(60):
        k = 0
        second_result = 0
        while k < 150:
            second_result += first_result_list[k] * second_weights_array[k][j]
            k += 1
        second_result_list.append(sigmoid(second_result))



    final_result_list = []
    j = 0
    for j in range(4):
        k = 0
        final_result = 0
        while k < 60:
            final_result += second_result_list[k] * third_weights_array[k][j]
            k += 1
        final_result_list.append(sigmoid(final_result))



    maximum_element_number = maximum_element_number_finder(final_result_list)
    if random_training_data[i][1][maximum_element_number] == 1:
        correct_result_counter += 1



print("The accuracy is: " + str(correct_result_counter / 2) + "%")











