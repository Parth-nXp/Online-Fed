import numpy as np
from dataset_generator import DatasetGenerator
from federated_learning import FederatedLearning

class ExperimentRunner:
    """
    A class to run multiple federated learning experiments, handling data generation, training, 
    and evaluating performance.

    Attributes:
        num_clients (int): Number of clients participating in the experiment.
        rff_dim (int): Dimensionality of the random Fourier features.
        feature_dim (int): Dimensionality of the input features.
        num_iterations (int): Number of federated learning iterations (rounds).
        num_participating_clients (int): Number of clients participating in each round.
        learning_rate (float): Learning rate for the local updates.
        num_epochs (int): Number of local training epochs for each client.
        independent_experiments (int): Number of independent experiments to run.
    """

    def __init__(self, num_clients, rff_dim, feature_dim, num_iterations, num_participating_clients, learning_rate, num_epochs, independent_experiments):
        """
        Initializes the ExperimentRunner with the specified parameters.

        Args:
            num_clients (int): Number of clients.
            rff_dim (int): Dimension of random Fourier features (RFF).
            feature_dim (int): Dimension of input features.
            num_iterations (int): Number of rounds/iterations for federated learning.
            num_participating_clients (int): Number of clients to participate per round.
            learning_rate (float): Learning rate for local updates.
            num_epochs (int): Number of local training epochs.
            independent_experiments (int): Number of independent experiments to run.
        """
        self.num_clients = num_clients
        self.rff_dim = rff_dim
        self.feature_dim = feature_dim
        self.num_iterations = num_iterations
        self.num_participating_clients = num_participating_clients
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.independent_experiments = independent_experiments

    def run(self):
        """
        Runs the federated learning experiments. For each experiment, it generates datasets, 
        performs federated learning, and evaluates the results.
        
        Returns:
            average_global_mse_history (numpy.ndarray): The average MSE history across all experiments.
        """
        all_global_mse_histories = []

        for experiment in range(self.independent_experiments):
            print(f"Running Experiment {experiment + 1}/{self.independent_experiments}")

            # Generate training and testing datasets
            dataset_gen = DatasetGenerator(self.num_clients, self.num_iterations, self.feature_dim, self.rff_dim)
            training_data = dataset_gen.generate_dataset()
            testing_data = dataset_gen.generate_dataset()

            # Initialize and run federated learning
            federated = FederatedLearning(self.num_clients, self.rff_dim, self.num_participating_clients, self.learning_rate, self.num_epochs, self.num_iterations)
            global_mse_history = federated.federated_averaging(training_data)

            # Store the global loss (MSE) history
            all_global_mse_histories.append(global_mse_history)

            # Print final global model weights
            federated.print_final_global_model()

        # Compute and return the average loss history across all experiments
        average_global_mse_history = np.mean(all_global_mse_histories, axis=0)
        return average_global_mse_history
