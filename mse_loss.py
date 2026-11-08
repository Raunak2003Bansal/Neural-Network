class LossFunction():

    @staticmethod
    def mse_loss(expected, predicted):
        """Calculate the Mean Squared Error (MSE) loss."""
        total_loss = 0
        for e,p in zip(expected, predicted):
            total_loss = total_loss + (e-p)**2
        return total_loss / len(expected)



expected = [1, 0, 1, 0]

predicted = [0.62, 0.35, 0.75, 0.20]

loss = LossFunction.mse_loss(
    expected,
    predicted
)

print("MSE Loss:", loss)