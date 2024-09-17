from experiment_runner import ExperimentRunner

def main():
    """
    Main function to initialize and run the federated learning experiments.
    The function sets up the parameters and invokes the ExperimentRunner to 
    perform the experiment and print the results.
    """
    experiment = ExperimentRunner(
        num_clients=100,                # Number of clients participating in federated learning
        rff_dim=200,                    # Random Fourier features dimension
        feature_dim=4,                  # Input feature dimension
        num_iterations=1000,            # Number of federated learning iterations (rounds)
        num_participating_clients=20,   # Number of clients participating in each round
        learning_rate=0.55,             # Learning rate for local updates
        num_epochs=10,                  # Number of local training epochs
        independent_experiments=1       # Number of independent experiments to run
    )

    # Run the experiment and get the global loss history
    average_global_mse_history = experiment.run()

if __name__ == "__main__":
    main()
