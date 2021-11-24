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
    return 1 / (1 + np.exp(-input_number))


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





# Output is calculated here
w1 = np.random.normal(size=(150, 102))
w2 = np.random.normal(size=(60, 150))
w3 = np.random.normal(size=(4, 60))
b1 = np.zeros((150, 1))
b2 = np.zeros((60, 1))
b3 = np.zeros((4, 1))
correct_result_counter = 0

correct_result_counter = 0
for train_data in random_training_data[:200]:
    a0 = train_data[0]
    a1 = sigmoid(w1 @ a0 + b1)
    a2 = sigmoid(w2 @ a1 + b2)
    a3 = sigmoid(w3 @ a2 + b3)

    model_output = np.where(a3 == np.amax(a3))
    real_output = np.where(train_data[1] == np.amax(train_data[1]))

    if model_output == real_output:
        correct_result_counter += 1

print("The accuracy is: " + str(correct_result_counter / 2) + "%")











