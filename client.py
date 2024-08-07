import numpy as np

class Client:
    def __init__(self, feature_dim: int, rff_dim: int, num_iterations: int):
        """
        Initialize a client with random weights and data.

        :param feature_dim: Dimension of the feature space.
        :param rff_dim: Dimension of the random Fourier features.
        :param num_iterations: Number of iterations for data generation.
        """
        self.feature_dim = feature_dim
        self.rff_dim = rff_dim
        self.num_iterations = num_iterations
        self.W = np.random.randn(feature_dim, rff_dim)
        self.b = np.random.uniform(0, 2 * np.pi, (1, rff_dim))
        self.x = np.zeros((num_iterations, feature_dim))
        self.y = np.zeros((num_iterations, 1))
        self.z = np.zeros((num_iterations, rff_dim))
        self.local_weights = np.random.randn(rff_dim)
        self.generate_data()
        self.calculate_rff()

    def generate_data(self):
        """
        Generate synthetic data for the client using a specified stochastic process.
        """
        theta_k = np.random.uniform(0.2, 0.9)
        mu_k = np.random.uniform(-0.2, 0.2)
        sigma2_uk = np.random.uniform(0.2, 1.2)
        sigma2_nuk = np.random.uniform(0.005, 0.03)
        uk = np.random.normal(mu_k, np.sqrt(sigma2_uk), (self.num_iterations, self.feature_dim))
        nuk = np.random.normal(0, np.sqrt(sigma2_nuk), (self.num_iterations, 1))

        self.x[0] = uk[0]
        for n in range(1, self.num_iterations):
            self.x[n] = theta_k * self.x[n - 1] + np.sqrt(1 - theta_k**2) * uk[n]
            self.y[n] = (np.sqrt(self.x[n, 0]**2 + np.sin(np.pi * self.x[n, 3])**2) + 
                         (0.8 - 0.5 * np.exp(-self.x[n, 1]**2) * self.x[n, 2])) + nuk[n]

    def calculate_rff(self):
        """
        Calculate random Fourier features for the client's data.
        """
        self.z = np.sqrt(2 / self.rff_dim) * np.cos(np.dot(self.x, self.W) + self.b)
