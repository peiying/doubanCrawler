import random
import numpy
import math

def sigmoid(t):
    return 1 / (1 + math.exp(-t))

def neuron_output(weights, inputs):
    return sigmoid(numpy.dot(weights, inputs))

def feed_forward(neural_network, input_vector):
    outputs = []

    for layer in neural_network:
        input_with_bias = input_vector + [1]
        output = [neuron_output(neuron, input_with_bias) for neuron in layer]
        outputs.append(output)

        input_vector = output
    return outputs

def backpropagate(network, input_vector, targets):
    hidden_outputs, outputs = feed_forward(network, input_vector)
    # the output * (1 - output) is from the derivative of sigmoid || zip: convert to matrix
    output_deltas = [output * (1 - output) * (output - target) for output, target in zip(outputs, targets)]

    # adjust weights for output layer, one neuron at a time
    for i, output_neuron in enumerate(network[-1]):
        # focus on the ith output layer neuron
        for j, hidden_output in enumerate(hidden_outputs + [1]):
            # adjust the jth weight based on both
            # this neuron's delta and its jth input
            output_neuron[j] -= output_deltas[i] * hidden_output

    # back-propagate errors to hidden layer
    hidden_deltas = [hidden_output * (1 - hidden_output) * numpy.dot(output_deltas, [n[i] for n in output_layer]) for i, hidden_output in enumerate(hidden_outputs)]
    # adjust weights for hidden layer, one neuron at a time
    for i, hidden_neuron in enumerate(network[0]):
        for j, input in enumerate(input_vector + [1]):
            hidden_neuron[j] -= hidden_deltas[i] * input

def predict(input):
    return feed_forward(network, input)[-1]

random.seed(0)
input_size = 25
num_hidden = 5
output_size = 10
#generate 5*26 matrix to represent hidden_layer's weights
hidden_layer = [[random.random() for __ in range(input_size + 1)] for __ in range(num_hidden)]
#generate 10*6 matrix to represent output_layer's weights
output_layer = [[random.random() for __ in range(num_hidden + 1)] for __ in range(output_size)]

network = [hidden_layer, output_layer]
for __ in range(10000):
    for input_vector, target_vector in zip(inputs, targets):
        backpropagate(network, input_vector, target_vector)

predict(inputs[7])
