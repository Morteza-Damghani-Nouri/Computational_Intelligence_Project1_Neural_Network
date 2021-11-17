% % time
# Allocate W matrix and vector b for each layer.

# Initialize W with random normal distribution for each layer.
W1 = np.random.normal(size=(16, NUMBER_OF_PIXELS))
W2 = np.random.normal(size=(16, 16))
W3 = np.random.normal(size=(10, 16))

# Initialize b = 0, for each layer.
b1 = np.zeros((16, 1))
b2 = np.zeros((16, 1))
b3 = np.zeros((10, 1))

total_costs = []
batches = [train_set[x:x + batch_size] for x in range(0, 100, batch_size)]
for epoch in range(number_of_epochs):
    for batch in batches:
        # allocate grad_W matrix for each layer
        grad_W1 = np.zeros((16, NUMBER_OF_PIXELS))
        grad_W2 = np.zeros((16, 16))
        grad_W3 = np.zeros((10, 16))
        # allocate grad_b for each layer
        grad_b1 = np.zeros((16, 1))
        grad_b2 = np.zeros((16, 1))
        grad_b3 = np.zeros((10, 1))

        for image, label in batch:
            # compute the output (image is equal to a0)
            a1 = sigmoid(W1 @ image + b1)
            a2 = sigmoid(W2 @ a1 + b2)
            a3 = sigmoid(W3 @ a2 + b3)

            # ---- Last layer
            # weight
            grad_W3 += (2 * (a3 - label) * a3 * (1 - a3)) @ np.transpose(a2)

            # bias
            grad_b3 += 2 * (a3 - label) * a3 * (1 - a3)

            # ---- 3rd layer
            # activation
            delta_3 = np.zeros((16, 1))
            delta_3 += np.transpose(W3) @ (2 * (a3 - label) * (a3 * (1 - a3)))

            # weight
            grad_W2 += (a2 * (1 - a2) * delta_3) @ np.transpose(a1)

            # bias
            grad_b2 += delta_3 * a2 * (1 - a2)

            # ---- 2nd layer
            # activation
            delta_2 = np.zeros((16, 1))
            delta_2 += np.transpose(W2) @ delta_3 * a2 * (1 - a2)

            # weight
            grad_W1 += (delta_2 * a1 * (1 - a1)) @ np.transpose(image)

            # bias
            grad_b1 += delta_2 * a1 * (1 - a1)

        W3 = W3 - (learning_rate * (grad_W3 / batch_size))
        W2 = W2 - (learning_rate * (grad_W2 / batch_size))
        W1 = W1 - (learning_rate * (grad_W1 / batch_size))

        b3 = b3 - (learning_rate * (grad_b3 / batch_size))
        b2 = b2 - (learning_rate * (grad_b2 / batch_size))
        b1 = b1 - (learning_rate * (grad_b1 / batch_size))

    # calculate cost average per epoch
    cost = 0
    for train_data in train_set[:100]:
        a0 = train_data[0]
        a1 = sigmoid(W1 @ a0 + b1)
        a2 = sigmoid(W2 @ a1 + b2)
        a3 = sigmoid(W3 @ a2 + b3)

        for j in range(10):
            cost += np.power((a3[j, 0] - train_data[1][j, 0]), 2)

    cost /= 100
    total_costs.append(cost)