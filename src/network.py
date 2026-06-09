import numpy as np

class StaticMLP:
    # Lightweight Multi-Layer Perceptron parameterizing the target measure.
    def __init__(self, input_dim, output_dim):
        self.W1 = np.random.randn(8, input_dim) * 0.1
        self.b1 = np.zeros(8)
        self.W2 = np.random.randn(output_dim, 8) * 0.1
        self.b2 = np.zeros(output_dim)

def forward_pass(mlp, obs):
    # Maps raw environmental observation to event-centric measure.
    # Layer 1: Dense + ReLU
    h1 = mlp.W1 @ obs + mlp.b1
    h1 = np.maximum(0.0, h1)

    # Layer 2: Output Mapping
    raw_out = mlp.W2 @ h1 + mlp.b2

    # Numerically stable Softmax
    raw_out -= np.max(raw_out)
    exp_out = np.exp(raw_out)
    return exp_out / np.sum(exp_out), h1

def backward_pass(
    mlp, obs, h1, output_probs, target_measure, learning_rate=0.01
):
    """
    Executes analytical gradient descent to train the neural network weights.
    Minimizes Cross-Entropy loss between current output and supervised target
    measure.
    """
    # 1. Compute Softmax output error gradient (dL / dz2)
    delta2 = output_probs - target_measure  # Shape: (output_dim,)

    # 2. Compute gradients for Output Layer (W2, b2)
    dW2 = np.outer(delta2, h1)      # Shape: (output_dim, 8)
    db2 = delta2                    # Shape: (output_dim,)

    # 3. Propagate error back through ReLU activation of Layer 1
    # Derivative of ReLU is 1 if input > 0 else 0
    relu_grad = (h1 > 0).astype(float)
    delta1 = (mlp.W2.T @ delta2) * relu_grad    # Shape: (8,)

    # 4. Compute gradients for Input Layer (W1, b1)
    dW1 = np.outer(delta1, obs)     # Shape: (8, input_dim)
    db1 = delta1                    # Shape: (8,)

    # 5. Gradient Descent Update Step
    mlp.W2 -= learning_rate * dW2
    mlp.b2 -= learning_rate * db2
    mlp.W1 -= learning_rate * dW1
    mlp.b1 -= learning_rate * db1