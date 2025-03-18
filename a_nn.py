import numpy as np 

np.random.seed(0)

class Neuron:
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.weights = np.random.random(num_inputs)
        self.bias = np.random.random()
        self.output = 0

    def calc_output(self, inputs):
        self.output = np.dot(self.weights, inputs) + self.bias

        return self.output
    
class Layer:
    def __init__(self, num_inputs, num_neurons):
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons

        # Creating the neurons
        self.neurons = []
        for _ in range(self.num_neurons):
            neuron = Neuron(self.num_inputs)
            self.neurons.append(neuron)

        self.outputs = []

    def gen_outputs(self, inputs):
        self.outputs = [neuron.calc_output(inputs) for neuron in self.neurons]

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

        layer_inputs = inputs
        for layer in self.layers:
            layer_output = layer.gen_outputs(layer_inputs)
            layer_inputs = layer_output

        return layer_inputs


inputs = [4, 3, 7]

nn = NeuralNetwork(3, 2, 4, 3)
print(nn.forward(inputs))
