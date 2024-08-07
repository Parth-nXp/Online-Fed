import numpy as np
import matplotlib.pyplot as plt
from federated_learning import federated_learning

def main():
    """
    Main function to perform federated learning and plot the results.
    """
    # Hyperparameters
    num_clients = 100
    independent_experiment = 500
    rff_dim = 200
    num_participating_clients = 40
    learning_rate = 0.75
    num_iterations = 1000

    mse_values_all_trials = np.zeros(num_iterations)

    for _ in range(independent_experiment):
        mse_values_all_trials += federated_learning(num_clients, num_participating_clients, rff_dim, learning_rate, num_iterations)

    mse_values_all_trials /= independent_experiment
    mse_values_all_trials /= max(mse_values_all_trials)

    mse_value_all_trials = 10 * np.log10(mse_values_all_trials)

    plt.plot(mse_value_all_trials)
    plt.xlabel("Iterations")
    plt.ylabel("MSE (dB)")
    plt.title("Mean Squared Error Over Iterations")
    plt.show()

if __name__ == "__main__":
    main()
