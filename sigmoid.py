import math

class Neuron():
    def __init__(self, weight1, weight2, bias):
        self.weight1 = weight1
        self.weight2 = weight2
        self.bias = bias

    def weighted_sum(self, input1, input2):
        """Calculate the weighted sum of inputs and bias."""
        return (input1 * self.weight1) + (input2 * self.weight2) + self.bias

    @staticmethod
    def activation(x):
        """Compute the sigmoid function."""
        return 1 / (1 + math.exp(-x))

    def predict(self, input1, input2):
        """Predict the output for given inputs."""
        z = self.weighted_sum(input1, input2)
        return self.activation(z)

if __name__ == "__main__":
    activation = Neuron(0.5, -0.3, 0.1)
    print(activation.predict(1, 0))