import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


training_inputs = np.array(
    [
        [0, 0, 1],
        [0, 0, 0],
        [0, 1, 0],
        [1, 1, 1],
        [1, 0, 1],
        [0, 1, 1],
    ]
)

training_outputs = np.array([[0, 0, 0, 1, 1, 0]]).T

np.random.seed(1)

synaptic_weight = 2 * np.random.random((3, 1)) - 1

print("Random weight")
print(synaptic_weight)

# Метод обратного распространения
for i in range(30000):
    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, synaptic_weight))

    err = training_outputs - outputs
    adjustments = np.dot(input_layer.T, err * (outputs * (1 - outputs)))

    synaptic_weight += adjustments


print("New weight")
print(synaptic_weight)

print("Result: ")
print(outputs)


new_inputs = np.array([[1, 1, 0], [0, 0, 1], [0, 0, 0]])
output = sigmoid(np.dot(new_inputs, synaptic_weight))
print("New inputs")
print(output)
