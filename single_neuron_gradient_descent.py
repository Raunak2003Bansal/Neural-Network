import math


class Neuron:
    """Utility methods for a sigmoid neuron."""

    @staticmethod
    def sigmoid(x: float) -> float:
        """Apply the sigmoid activation function."""
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def derivative_sigmoid(output: float) -> float:
        """Return the derivative of the sigmoid activation."""
        return output * (1.0 - output)


class GradientCalculator:
    """Compute gradients for weight updates."""

    @staticmethod
    def calculate_weight_gradient(
        expected: float,
        predicted: float,
        hidden_output: float,
    ) -> float:
        """Calculate the gradient for updating the weight."""
        loss_derivative = 2.0 * (predicted - expected)
        activation_derivative = Neuron.derivative_sigmoid(predicted)
        hidden_derivative = hidden_output
        return loss_derivative * activation_derivative * hidden_derivative


    @staticmethod
    def calculate_output_bias_gradient(expected, predicted):

        loss_gradient = 2 * (predicted - expected)

        sigmoid_gradient = Neuron.derivative_sigmoid(predicted)

        weighted_sum_gradient = 1

        return (
            loss_gradient
            * sigmoid_gradient
            * weighted_sum_gradient
        )


def main() -> None:
    """Run the single-neuron gradient descent example."""
    hidden_output = 0.6
    weight = 0.5
    bias = 0.1
    expected = 1.0
    learning_rate = 0.1

    z = weight * hidden_output + bias
    predicted = Neuron.sigmoid(z)

    gradient = GradientCalculator.calculate_weight_gradient(
        expected,
        predicted,
        hidden_output,
    )

    new_weight = weight - learning_rate * gradient

    print(f"Prediction: {predicted}")
    print(f"Gradient: {gradient}")
    print(f"Old weight: {weight}")
    print(f"New weight: {new_weight}")

    bias_gradient = GradientCalculator.calculate_output_bias_gradient(
    expected,
    predicted
)

    new_bias = bias - learning_rate * bias_gradient

    print("Bias gradient:", bias_gradient)
    print("Old bias:", bias)
    print("New bias:", new_bias)


if __name__ == "__main__":
    main()
