class ArtificalNeuron():
    """A two-input perceptron with a binary step activation.

    Args:
        w1: Initial weight for the first input.
        w2: Initial weight for the second input.
        bias: Initial bias added to the weighted sum.
        learning_rate: Step size used to update the weights and bias.
    """

    def __init__(self, w1 = 0.0, w2 = 0.0, bias = 0.0, learning_rate = 0.1):
        """Initialize the perceptron's weights, bias, and learning rate.

        Args:
            w1: Initial weight for the first input.
            w2: Initial weight for the second input.
            bias: Initial bias added to the weighted sum.
            learning_rate: Step size used during training.
        """
        self.w1 = w1
        self.w2 = w2
        self.bias = bias
        self.learning_rate = learning_rate

    def weighted_sum(self, x1, x2):
        """Calculate the weighted sum for a pair of inputs.

        Args:
            x1: First input value.
            x2: Second input value.

        Returns:
            The weighted sum of the inputs and the bias.
        """
        return x1*self.w1 + x2*self.w2 + self.bias

    def activation(self, z):
        """Apply the binary step activation function.

        Args:
            z: Weighted sum to classify.

        Returns:
            1 if ``z`` is non-negative; otherwise, 0.
        """
        return 1 if z >= 0 else 0

    def predict(self, x1, x2):
        """Predict a binary output for two input values.

        Args:
            x1: First input value.
            x2: Second input value.

        Returns:
            The activated weighted sum, either 0 or 1.
        """
        z = self.weighted_sum(x1, x2)
        return self.activation(z)

    def train(self, x1, x2, expected_output):
        """Update the perceptron's parameters using one labeled example.

        Args:
            x1: First input value.
            x2: Second input value.
            expected_output: Desired binary output for the input pair.

        Returns:
            The prediction error, calculated as expected minus predicted.

        Notes:
            Updates the weights and bias in place using the perceptron
            learning rule.
        """
        predicted_output = self.predict(x1, x2)
        error = expected_output - predicted_output
        self.w1 = self.w1 + error*self.learning_rate*x1
        self.w2 = self.w2 + error*self.learning_rate*x2
        self.bias = self.bias + error*self.learning_rate
        return error


def main():
    """Train on XOR examples and print errors, predictions, and parameters."""

    network = ArtificalNeuron()

    sample_data = [
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
    ]

    epochs = 20
    for epoch in range(epochs):
        total_error = 0
        for x1, x2, expected_output in sample_data:
            error = network.train(x1, x2, expected_output)
            total_error += abs(error)  

        print(f"Epoch {epoch}: Total Error: {total_error}")

        if total_error == 0:
            print(f"Training completed at epoch {epoch}")
            break


    for x1, x2, expected_output in sample_data:
        predicted_output = network.predict(x1, x2)
        print(f"Input: ({x1}, {x2}), Predicted Output: {predicted_output}, Expected Output: {expected_output}")

    print("\nLearned parameters:")
    print("w1:", network.w1)
    print("w2:", network.w2)
    print("Bias:", network.bias)


if __name__ == "__main__":
    main()
