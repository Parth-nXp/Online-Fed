import numpy as np
from client import Client

def federated_learning(num_clients: int, num_participating_clients: int, rff_dim: int, learning_rate: float, num_iterations: int) -> np.ndarray:
    """
    Perform federated learning across multiple clients.

    :param num_clients: Total number of clients.
    :param num_participating_clients: Number of clients participating in each round.
    :param rff_dim: Dimension of the random Fourier features.
    :param learning_rate: Learning rate for weight updates.
    :param num_iterations: Number of iterations for training.
    :return: Mean squared error values for all trials.
    """
    global_weights = np.random.randn(rff_dim)
    clients = [Client(feature_dim=5, rff_dim=rff_dim, num_iterations=num_iterations) for _ in range(num_clients)]
    mse_values_per_iteration = np.zeros(num_iterations)

    for n in range(num_iterations):
        selected_indices = np.random.choice(num_clients, num_participating_clients, replace=False)
        mse_per_iteration = 0

        for k in range(num_clients):
            epsilon = clients[k].y[n] - np.dot(clients[k].local_weights, clients[k].z[n])
            clients[k].local_weights += learning_rate * clients[k].z[n] * epsilon
            mse_per_iteration += epsilon**2

        mse_values_per_iteration[n] = mse_per_iteration / num_clients

        global_weights = np.zeros(rff_dim)
        for idx in selected_indices:
            global_weights += clients[idx].local_weights

        global_weights /= num_participating_clients

    return mse_values_per_iteration
