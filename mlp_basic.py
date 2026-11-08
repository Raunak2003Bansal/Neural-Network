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
    neuron1 = Neuron(0.5, -0.3, 0.1)
    neuron2 = Neuron(-0.4, 0.6, -0.2)
    x1= 1
    x2= 0
    hidden_output1 = neuron1.predict(x1, x2)
    hidden_output2 = neuron2.predict(x1, x2)

    print(f"Hidden Layer Outputs: Neuron 1: {hidden_output1}, Neuron 2: {hidden_output2}")

    neuron3 = Neuron(0.7, 0.4, -0.1)
    final_output = neuron3.predict(hidden_output1, hidden_output2)
    print(f"Final Output: {final_output}")