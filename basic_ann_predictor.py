import random


class BasicPredictor:
    """
    Predict a binary output from two inputs using a weighted sum and bias.

    Args:
        w1: Weight applied to the first input.
        w2: Weight applied to the second input.
        bias: Bias added to the weighted sum.
    """

    def __init__(self, w1: float = 0, w2: float = 0, bias: float = 1) -> None:
        self.w1 = w1
        self.w2 = w2
        self.bias = bias

    def weighted_sum(self, x1: float, x2: float) -> float:
        """Return the weighted sum of two inputs and the bias.

        Args:
            x1: First input value.
            x2: Second input value.

        Returns:
            The weighted sum before activation.
        """
        return x1 * self.w1 + x2 * self.w2 + self.bias

    def activation(self, z: float) -> int:
        """Convert a weighted sum to a binary output.

        Args:
            z: Weighted sum to classify.

        Returns:
            1 when ``z`` is non-negative; otherwise, 0.
        """
        return 1 if z >= 0 else 0

    def predict(self, x1: float, x2: float) -> int:
        """Predict the binary output for two input values.

        Args:
            x1: First input value.
            x2: Second input value.

        Returns:
            The activated weighted sum, either 0 or 1.
        """
        z = self.weighted_sum(x1, x2)
        return self.activation(z)



def main():
    """Generate predictions for random inputs and print the results."""
    w1 = random.uniform(-1,1)
    w2 = random.uniform(-1,1)
    bias = random.uniform(-1,1)
    predictor = BasicPredictor(w1, w2, bias)
    predictions = []
    for i in range(0, 100):
        x1 = random.random()
        x2 = random.random()
        y_hat = predictor.predict(x1, x2)
        predictions.append(y_hat)
    print(predictions)
    return



if __name__ == "__main__":
    main()