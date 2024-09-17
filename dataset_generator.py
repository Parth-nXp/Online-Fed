import numpy as np

class DatasetGenerator:
    """
    A class to generate synthetic datasets for training and testing in a federated learning setup.

    Attributes:
        num_clients (int): Number of clients participating in the federated learning.
        num_iterations (int): Number of iterations each client performs.
        feature_dim (int): Dimensionality of the input features.
        rff_dim (int): Dimensionality of the random Fourier features.
    """

    def __init__(self, num_clients, num_iterations, feature_dim, rff_dim):
        """
        Initializes the DatasetGenerator with the specified parameters.

        Args:
            num_clients (int): Number of clients.
            num_iterations (int): Number of iterations.
            feature_dim (int): Dimension of input features.
            rff_dim (int): Dimension of random Fourier features (RFF).
        """
        self.num_clients = num_clients
        self.num_iterations = num_iterations
        self.feature_dim = feature_dim
        self.rff_dim = rff_dim

    def generate_dataset(self):
        """
        Generates synthetic data for federated learning. Each client has input-output pairs.

        Returns:
            client_data (list): A list containing tuples of input (z_k) and output (y_k) data
            for each client, where z_k is the RFF representation and y_k is the output label.
        """
        # Generate input noise from a Laplace distribution
        v_k = np.random.laplace(0, 1, (self.num_iterations, self.feature_dim))

        # Initialize the input-output matrices
        x = np.zeros((self.num_clients, self.num_iterations, self.feature_dim))
        z = np.zeros((self.num_clients, self.num_iterations, self.rff_dim))
        y = np.zeros((self.num_clients, self.num_iterations, 1))

        # Random weights and bias for random Fourier features (RFF)
        W = np.random.randn(self.num_clients, self.feature_dim, self.rff_dim)
        b = np.random.uniform(0, 2 * np.pi, (self.num_clients, 1, self.rff_dim))

        # Noise generation based on a mixture of Gaussians
        eta_k = np.random.binomial(1, 0.5, self.num_iterations)
        sigma1 = 0.02
        sigma2 = 0.05
        nuk = eta_k[:, None] * np.random.normal(0, sigma1, (self.num_iterations, 1)) + \
              (1 - eta_k[:, None]) * np.random.normal(0, sigma2, (self.num_iterations, 1))

        # Parameters for the autoregressive (AR) process
        gamma_k = np.random.uniform(0.4, 0.7)
        delta_k = np.random.uniform(0.2, 0.6)
        lambda_k = 0.5

        # Generate synthetic data for each client
        x[:, 0] = v_k[0]  # Initial input
        for k in range(1, self.num_clients):
            for n in range(2, self.num_iterations):
                x[k, n, :] = gamma_k * x[k, n-2, :] + delta_k * np.tanh(x[k, n-1, :]) + lambda_k * v_k[n]

        # Map input x to RFF z and generate output y
        for k in range(1, self.num_clients):
            z[k, :, :] = np.sqrt(2 / self.rff_dim) * np.cos(np.dot(x[k, :, :], W[k, :, :]) + b[k, :, :])
            for n in range(1, self.num_iterations):
                y[k, n, :] = np.log(1 + np.abs(x[k, n, 0]) * np.cos(np.pi * x[k, n, 1])) + \
                             (1 + np.exp(-x[k, n, 3]**2)) * x[k, n, 2] + nuk[n]

        # Package the data for each client
        client_data = []
        for k in range(self.num_clients):
            z_k = z[k, -1, :].reshape(-1, 1)  # Reshape RFF to 200x1
            y_k = y[k, -1, :].reshape(-1, 1)  # Reshape output to 1x1
            client_data.append((z_k, y_k))

        return client_data
