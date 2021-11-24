import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time

NUMBER_OF_PIXELS = 102
NUMBER_OF_EPOCHS = 20
BATCH_SIZE = 10
LEARNING_RATE = 1



# This function receives an input and returns the sigmoid amount of the input
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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
    desired_output = np.array([0, 0, 0, 0])
    desired_output[int(train_set_labels[i])] = 1
    desired_output = desired_output.reshape(4, 1)
    train_set.append((train_set_features[i].reshape(102, 1), desired_output))

for i in range(len(test_set_features)):
    desired_output = np.array([0, 0, 0, 0])
    desired_output[int(test_set_labels[i])] = 1
    desired_output = desired_output.reshape(4, 1)
    test_set.append((test_set_features[i].reshape(102, 1), desired_output))

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




# Main part of the model starts here
# Random weights and biases are generated here
start = time.time()
w1 = np.random.normal(size=(150, NUMBER_OF_PIXELS))
w2 = np.random.normal(size=(60, 150))
w3 = np.random.normal(size=(4, 60))

b1 = np.zeros((150, 1))
b2 = np.zeros((60, 1))
b3 = np.zeros((4, 1))

costs_list = []

for epoch in range(NUMBER_OF_EPOCHS):
    batches = [random_training_data[element:element + BATCH_SIZE] for element in range(0, 200, BATCH_SIZE)]
    for batch in batches:

        grad_w1 = np.zeros((150, NUMBER_OF_PIXELS))
        grad_w2 = np.zeros((60, 150))
        grad_w3 = np.zeros((4, 60))

        grad_b1 = np.zeros((150, 1))
        grad_b2 = np.zeros((60, 1))
        grad_b3 = np.zeros((4, 1))

        for input_data, desired_output in batch:
            a1 = sigmoid(w1 @ input_data + b1)
            a2 = sigmoid(w2 @ a1 + b2)
            a3 = sigmoid(w3 @ a2 + b3)
            grad_w3 += (2 * (a3 - desired_output) * a3 * (1 - a3)) @ np.transpose(a2)
            grad_b3 += 2 * (a3 - desired_output) * a3 * (1 - a3)
            cost_ak_rond = np.zeros((60, 1))
            cost_ak_rond += np.transpose(w3) @ (2 * (a3 - desired_output) * (a3 * (1 - a3)))
            grad_w2 += (a2 * (1 - a2) * cost_ak_rond) @ np.transpose(a1)
            grad_b2 += cost_ak_rond * a2 * (1 - a2)
            cost_am_rond = np.zeros((150, 1))
            cost_am_rond += np.transpose(w2) @ (cost_ak_rond * a2 * (1 - a2))
            grad_w1 += (cost_am_rond * a1 * (1 - a1)) @ np.transpose(input_data)
            grad_b1 += cost_am_rond * a1 * (1 - a1)



        w3 = w3 - (LEARNING_RATE * (grad_w3 / BATCH_SIZE))
        w2 = w2 - (LEARNING_RATE * (grad_w2 / BATCH_SIZE))
        w1 = w1 - (LEARNING_RATE * (grad_w1 / BATCH_SIZE))
        b3 = b3 - (LEARNING_RATE * (grad_b3 / BATCH_SIZE))
        b2 = b2 - (LEARNING_RATE * (grad_b2 / BATCH_SIZE))
        b1 = b1 - (LEARNING_RATE * (grad_b1 / BATCH_SIZE))

    # Cost is calculated here
    cost = 0
    for train_data in random_training_data[:200]:
        a0 = train_data[0]
        a1 = sigmoid(w1 @ a0 + b1)
        a2 = sigmoid(w2 @ a1 + b2)
        a3 = sigmoid(w3 @ a2 + b3)

        for j in range(4):
            cost += np.power((a3[j, 0] - train_data[1][j, 0]), 2)

    cost /= 200
    costs_list.append(cost)


stop = time.time()
epoch_size = [x for x in range(20)]
plt.plot(epoch_size, costs_list)
plt.show()
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

print("The accuracy is: " + str(round((correct_result_counter / 200) * 100, 2)) + "%")
print("The executation time is: " + str(round((stop - start), 2)) + " seconds")

