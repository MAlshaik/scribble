# List of inputs

inputs = [1.0, 2.0, 3.0]

# Corresponding weights

weights = [0, 0, 0]

# Bias

bias = 0.0

# Calculate the dot product

output = 0

for i, weight in zip(inputs, weights):

    output += weight*i

output += bias

print("Output:", output)
