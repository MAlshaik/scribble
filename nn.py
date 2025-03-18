import numpy as np 

class Neuron:
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()
        self.output = 0


    def forward(self, inputs):
        self.output = 0

        for input, weight in zip(inputs, self.weights):
            self.output += input * weight

        self.output += self.bias

        return self.output
    

