import numpy as np
import unittest
from nn import Neuron, Layer, NeuralNetwork

class TestNeuralNetwork(unittest.TestCase):
    def test_neuron_initialization(self):
        """Test that a neuron initializes with the correct number of weights."""
        num_inputs = 5
        neuron = Neuron(num_inputs)
        self.assertEqual(len(neuron.weights), num_inputs)
        self.assertIsInstance(neuron.bias, float)
        
    def test_neuron_forward(self):
        """Test that a neuron's forward pass calculates the correct output."""
        num_inputs = 3
        neuron = Neuron(num_inputs)
        # Set fixed weights and bias for deterministic testing
        neuron.weights = np.array([0.5, -0.2, 0.1])
        neuron.bias = 0.4
        inputs = np.array([1.0, 2.0, 3.0])
        
        expected_output = (0.5 * 1.0) + (-0.2 * 2.0) + (0.1 * 3.0) + 0.4
        output = neuron.forward(inputs)
        
        self.assertAlmostEqual(output, expected_output)
        
    def test_layer_initialization(self):
        """Test that a layer initializes with the correct number of neurons."""
        num_inputs = 4
        num_neurons = 3
        layer = Layer(num_inputs, num_neurons)
        
        self.assertEqual(len(layer.neurons), num_neurons)
        for neuron in layer.neurons:
            self.assertEqual(len(neuron.weights), num_inputs)
        
    def test_layer_forward(self):
        """Test that a layer's forward pass calculates the correct outputs."""
        num_inputs = 3
        num_neurons = 2
        layer = Layer(num_inputs, num_neurons)
        
        # Set fixed weights and biases for deterministic testing
        for i, neuron in enumerate(layer.neurons):
            neuron.weights = np.array([0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1)])
            neuron.bias = 0.1 * (i + 1)
        
        inputs = np.array([0.5, 1.0, 1.5])
        
        layer.forward(inputs)
        
        # Calculate expected outputs
        expected_outputs = []
        for i, neuron in enumerate(layer.neurons):
            expected = (0.1 * (i + 1) * 0.5) + (0.2 * (i + 1) * 1.0) + (0.3 * (i + 1) * 1.5) + (0.1 * (i + 1))
            expected_outputs.append(expected)
        
        self.assertEqual(len(layer.outputs), num_neurons)
        for actual, expected in zip(layer.outputs, expected_outputs):
            self.assertAlmostEqual(actual, expected)
    
    def test_neural_network_initialization(self):
        """Test that a neural network initializes with the correct number of layers."""
        num_inputs = 4
        num_hidden_layers = 2
        num_hidden_layer_neurons = 3
        num_outputs = 2
        
        nn = NeuralNetwork(num_inputs, num_hidden_layers, num_hidden_layer_neurons, num_outputs)
        
        # Should have hidden layers + output layer
        self.assertEqual(len(nn.layers), num_hidden_layers + 1)
        
        # Check hidden layers
        for i in range(num_hidden_layers):
            self.assertEqual(len(nn.layers[i].neurons), num_hidden_layer_neurons)
            
        # Check output layer
        self.assertEqual(len(nn.layers[-1].neurons), num_outputs)
    
    def test_neural_network_forward(self):
        """Test that a neural network's forward pass correctly propagates inputs through all layers."""
        num_inputs = 2
        num_hidden_layers = 2
        num_hidden_layer_neurons = 2
        num_outputs = 2
        
        nn = NeuralNetwork(num_inputs, num_hidden_layers, num_hidden_layer_neurons, num_outputs)
        
        # Set fixed weights and biases for deterministic testing
        # First hidden layer
        nn.layers[0].neurons[0].weights = np.array([0.15, 0.20])
        nn.layers[0].neurons[0].bias = 0.35
        nn.layers[0].neurons[1].weights = np.array([0.25, 0.30])
        nn.layers[0].neurons[1].bias = 0.35
        
        # Second hidden layer
        nn.layers[1].neurons[0].weights = np.array([0.40, 0.45])
        nn.layers[1].neurons[0].bias = 0.60
        nn.layers[1].neurons[1].weights = np.array([0.50, 0.55])
        nn.layers[1].neurons[1].bias = 0.60
        
        # Inputs
        inputs = np.array([0.05, 0.10])
        
        # Manual calculation of expected output
        # First layer outputs
        h1_1 = 0.15 * 0.05 + 0.20 * 0.10 + 0.35  # = 0.3775
        h1_2 = 0.25 * 0.05 + 0.30 * 0.10 + 0.35  # = 0.3925
        
        # Second layer outputs
        h2_1 = 0.40 * h1_1 + 0.45 * h1_2 + 0.60  # = 0.92763
        h2_2 = 0.50 * h1_1 + 0.55 * h1_2 + 0.60  # = 0.98513
        
        # Get the actual output
        output = nn.forward(inputs)
        
        # Check that outputs match our expected calculations
        self.assertAlmostEqual(output[0], h2_1, places=4)
        self.assertAlmostEqual(output[1], h2_2, places=4)

    def test_integration(self):
        """Integration test with a small network."""
        # Create a small network: 2 inputs, 1 hidden layer with 2 neurons, 1 output
        nn = NeuralNetwork(2, 1, 2, 1)
        
        # Hidden layer
        nn.layers[0].neurons[0].weights = np.array([0.15, 0.20])
        nn.layers[0].neurons[0].bias = 0.35
        nn.layers[0].neurons[1].weights = np.array([0.25, 0.30])
        nn.layers[0].neurons[1].bias = 0.35
        
        # Output layer
        nn.layers[1].neurons[0].weights = np.array([0.40, 0.45])
        nn.layers[1].neurons[0].bias = 0.60
        
        # Inputs
        inputs = np.array([0.05, 0.10])
        
        # Forward pass
        output = nn.forward(inputs)
        
        # Expected output calculation:
        # Hidden layer, neuron 1: 0.15*0.05 + 0.20*0.10 + 0.35 = 0.3775
        # Hidden layer, neuron 2: 0.25*0.05 + 0.30*0.10 + 0.35 = 0.3925
        # Output layer: 0.40*0.3775 + 0.45*0.3925 + 0.60 = 0.92763
        
        expected_output = 0.92763
        
        # Check output
        self.assertAlmostEqual(output[0], expected_output, places=4)

if __name__ == "__main__":
    unittest.main()
