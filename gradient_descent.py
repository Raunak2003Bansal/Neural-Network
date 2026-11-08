"""Simple gradient descent example for updating weights."""


class GradientDescent:
    """Utility class for a basic gradient descent update."""

    @staticmethod
    def update_weight(weight: float, gradient: float, learning_rate: float) -> float:
        """Return the updated weight after a descent step."""
        return weight - learning_rate * gradient


def main() -> None:
    """Run the example calculation."""
    weight = 0.5
    gradient = 0.8
    learning_rate = 0.1

    new_weight = GradientDescent.update_weight(
        weight,
        gradient,
        learning_rate,
    )

    print(f"Old weight: {weight}")
    print(f"New weight: {new_weight}")


if __name__ == "__main__":
    main()
