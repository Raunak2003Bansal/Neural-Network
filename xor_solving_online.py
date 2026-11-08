import random
import math
random.seed(7)




class NeuralNetwork:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

        # Input → Hidden layer weights
        self.w_x11 = 0.0
        self.w_x21 = 0.1
        self.w_x12 = 0.2
        self.w_x22 = 0.3

        # Hidden → Output layer weights
        self.w_h11 = 0.4
        self.w_h21 = 0.5

        # Biases
        self.bias_h1 = 0.1
        self.bias_h2 = 0.2
        self.bias_o1 = 0.3


    def sigmoid(self, z):
        return 1/(1+ math.exp(-z))

    
    def forward(self, x1, x2):
        # Hidden neuron 1
        z1 = (
            x1 * self.w_x11
            + x2 * self.w_x21
            + self.bias_h1
        )
        h1 = self.sigmoid(z1)

        # Hidden neuron 2
        z2 = (
            x1 * self.w_x12
            + x2 * self.w_x22
            + self.bias_h2
        )
        h2 = self.sigmoid(z2)

        # Output neuron
        z0 = (
            h1 * self.w_h11
            + h2 * self.w_h21
            + self.bias_o1
        )
        o1 = self.sigmoid(z0)

        return h1, h2, o1

    def calculate_loss(self, target, predicted):
        return 0.5 * (predicted - target) ** 2


    def calculate_deltas(self, h1, h2, o1, target):
        del_o1 = (o1 - target) * (1 - o1) * o1
        del_h1 = del_o1 * self.w_h11 * (1 - h1) * h1
        del_h2 = del_o1 * self.w_h21 * (1 - h2) * h2
        return del_o1, del_h1, del_h2

    def calculate_gradients(self, delta, input_):
        gradient = delta * input_
        return gradient

    def update_weights(self, weight, gradient):
        weight -= gradient*self.learning_rate
        return weight

    def step_function(self, o1):
        return 1 if o1>=0.5 else 0


    def train(self, epochs=100000):
        inputs = [[0,0,0],[0,1,1],[1,0,1],[1,1,0]]
        for epoch in range(epochs):
            for input_ in inputs:
                h1, h2, o1 = self.forward(input_[0], input_[1])
                loss = self.calculate_loss(input_[2], o1)

                del_o1, del_h1, del_h2 = self.calculate_deltas(h1, h2, o1, input_[2])
                gradient_w_h11 = self.calculate_gradients(del_o1, h1)
                gradient_w_h21 = self.calculate_gradients(del_o1, h2)
                gradient_w_x11 = self.calculate_gradients(del_h1, input_[0])
                gradient_w_x21 = self.calculate_gradients(del_h1, input_[1])
                gradient_w_x12 = self.calculate_gradients(del_h2, input_[0])
                gradient_w_x22 = self.calculate_gradients(del_h2, input_[1])

                self.w_h11 = self.update_weights(self.w_h11, gradient_w_h11)
                self.w_h21 = self.update_weights(self.w_h21, gradient_w_h21)
                self.w_x11 = self.update_weights(self.w_x11, gradient_w_x11)
                self.w_x21 = self.update_weights(self.w_x21, gradient_w_x21)
                self.w_x12 = self.update_weights(self.w_x12, gradient_w_x12)
                self.w_x22 = self.update_weights(self.w_x22, gradient_w_x22)

                self.bias_o1 -= del_o1 * self.learning_rate
                self.bias_h1 -= del_h1 * self.learning_rate
                self.bias_h2 -= del_h2 * self.learning_rate

        return
    
        
def main():
    inputs = [[0,0,0],[0,1,1],[1,0,1],[1,1,0]]
    print("\nFinal XOR predictions:")
    network = NeuralNetwork()
    network.train()
    for input_ in inputs:
        h1, h2, o1 = network.forward(input_[0], input_[1])
        predicted = network.step_function(o1)
        print(f"Input: ({input_[0]},{input_[1]}) | Target: {input_[2]} | Output: {o1} | Prediction: {predicted}")




if __name__=="__main__":
    main()