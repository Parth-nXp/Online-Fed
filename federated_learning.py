import numpy as np

class FederatedLearning:
    """
    A class to perform federated learning using synthetic datasets.

    Attributes:
        num_clients (int): Number of clients participating in the federated learning.
        rff_dim (int): Dimensionality of the random Fourier features.
        num_participating_clients (int): Number of clients participating in each round of federated learning.
        learning_rate (float): Learning rate for the local gradient updates.
        num_epochs (int): Number of local training epochs each client performs per round.
        num_iterations (int): Total number of federated learning iterations (rounds).
    """

    def __init__(self, num_clients, rff_dim, num_participating_clients, learning_rate, num_epochs, num_iterations):
        """
        Initializes the FederatedLearning with the specified parameters and
        creates an initial random global model.
        
        Args:
            num_clients (int): Number of clients.
            rff_dim (int): Dimension of random Fourier features (RFF).
            num_participating_clients (int): Number of clients to participate per round.
            learning_rate (float): Learning rate for local updates.
            num_epochs (int): Number of local training epochs.
            num_iterations (int): Number of rounds/iterations for federated learning.
        """
        self.num_clients = num_clients
        self.rff_dim = rff_dim
        self.num_participating_clients = num_participating_clients
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_iterations = num_iterations
        self.global_fc = np.random.randn(rff_dim, 1)  # Initial random global model weights

    def client_update(self, client_data, global_fc):
        """
        Performs local model training for a single client.

        Args:
            client_data (tuple): A tuple containing the input and output data for the client.
            global_fc (numpy.ndarray): The current global model parameters.

        Returns:
            fc (numpy.ndarray): The locally updated model parameters for the client.
        """
        fc = global_fc.copy()  # Copy global model parameters
        inputs, labels = client_data
        inputs = inputs.reshape(1, -1)  # Reshape input for matrix operations
        
        # Perform local training
        for epoch in range(self.num_epochs):
            outputs = np.dot(inputs, fc)  # Forward pass
            loss = np.mean((outputs - labels) ** 2)  # Calculate loss (MSE)
            grad = 2 * np.dot(inputs.T, outputs - labels)  # Compute gradients
            fc -= self.learning_rate * grad  # Update local model parameters
        return fc

    def federated_averaging(self, training_data):
        """
        Conducts federated averaging by selecting clients and updating the global model.

        Args:
            training_data (list): A list of training data for all clients.

        Returns:
            global_mse_history (list): A list of global loss (MSE) values across all rounds.
        """
        global_mse_history = []
        
        # Iterate through the number of federated learning rounds
        for round in range(self.num_iterations):
            # Randomly select a subset of clients for this round
            selected_clients = np.random.choice(range(len(training_data)), size=self.num_participating_clients, replace=False)
            
            # Perform local training and collect the updated models
            client_models = [self.client_update(training_data[client_index], self.global_fc) for client_index in selected_clients]

            # Aggregate the models (federated averaging)
            aggregated_fc = np.zeros_like(self.global_fc)
            for client_fc in client_models:
                aggregated_fc += client_fc
            self.global_fc = aggregated_fc / len(client_models)  # Update global model

            # Evaluate the updated global model (optional, could be testing data)
            global_mse = self.evaluate_global_model(training_data)
            global_mse_history.append(global_mse)
        
        return global_mse_history

    def evaluate_global_model(self, testing_data):
        """
        Evaluates the global model on a test dataset to compute the global MSE.

        Args:
            testing_data (list): A list of testing data for evaluation.

        Returns:
            global_mse (float): The computed mean squared error across all test clients.
        """
        global_mse = 0
        for inputs, labels in testing_data:
            inputs = inputs.reshape(1, -1)  # Reshape input for matrix operations
            outputs = np.dot(inputs, self.global_fc)  # Forward pass using global model
            global_mse += np.mean((outputs - labels) ** 2)  # Compute MSE
        return global_mse / len(testing_data)

    def print_final_global_model(self):
        """Prints the final global model parameters (weights) after training."""
        print("Final Global Model Parameters (Weights):")
        print(self.global_fc)
