import numpy as np 

np.random.seed(0)

class Neuron:
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.weights = np.random.random(num_inputs)
        self.bias = np.random.random()
        self.output = 0



    def forward(self, inputs):
        self.output = 0

        for input, weight in zip(inputs, self.weights):
            self.output += input * weight

        self.output += self.bias

        return self.output
    


class Layer:

    def __init__(self, num_inputs, num_neurons):

        self.num_inputs = num_inputs

        self.num_neurons = num_neurons

        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]

        self.outputs = []

    def forward(self, inputs):
        self.outputs = [neuron.forward(inputs) for neuron in self.neurons]

        return self.outputs



class NeuralNetwork:

    def __init__(self, num_inputs, num_hidden_layers, num_hidden_layer_neurons, num_output_layer_neurons):
        self.num_inputs = num_inputs
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_layer_neurons = num_hidden_layer_neurons
        self.num_output_layer_neurons = num_output_layer_neurons

        # Creating the hidden layers
        self.layers = []
        input_size = num_inputs

        for _ in range(num_hidden_layers):
            layer = Layer(input_size, self.num_hidden_layer_neurons)
            self.layers.append(layer)
            input_size = self.num_hidden_layer_neurons

        # Create and add the output layer, the last layer
        output_layer = Layer(num_inputs=input_size, num_neurons=num_output_layer_neurons)
        self.layers.append(output_layer)



    def forward(self, inputs):

        # Take the inputs and pass those inputs to each layer in the network

        # Tip, use a for loop and one variable to keep track of the outputs of a single layer

        # Keep updating that single variable with the outputs of the layers

        # At the end, whatever is in that variable will be the output of the last layer
        output = inputs
        for layer in self.layers:
            layer.forward(output)
            output = layer.outputs

        return output


inputs = [4, 3, 7]

nn = NeuralNetwork(3, 2, 4, 3)
print(nn.forward(inputs))
