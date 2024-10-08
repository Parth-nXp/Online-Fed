# Online Federated Learning with Communication Efficiency

This project implements online federated learning strategies for resource-constrained devices using random Fourier features (RFFs) for kernel regression. The model training is performed in real-time with continuous data streams, ensuring efficient local model updates and maintaining model performance.

This simulation is inspired by the work of Anthony Kuh, titled "Real-Time Kernel Learning for Sensor Networks using Principles of Federated Learning," presented at the APSIPA Annual Summit and Conference 2021 in Tokyo, Japan. [Read the paper](https://ieeexplore.ieee.org/document/9689337).

## Project Structure

The project is divided into four main scripts:

1. `dataset_generator.py`
   - **Purpose**: Contains the `DatasetGenerator` class, which generates synthetic data for federated learning. Each client’s input-output data is mapped to random Fourier features (RFF), and a corresponding label is generated.
   - **Key Functionality**:
       - `generate_dataset()`: Generates input-output pairs for each client using random Fourier features and an autoregressive model for the input.

2. `federated_learning.py`
   - **Purpose**: Contains the `FederatedLearning` class, which implements the local training updates for each client and performs federated averaging to update the global model.
   - **Key Functionality**:
      - `client_update()`: Performs local updates on each client’s data by computing gradients and adjusting local model weights.
      - `federated_averaging()`: Aggregates the locally updated models from selected clients to update the global model using federated averaging.
      - `evaluate_global_model()`: Evaluates the global model’s performance on a dataset by calculating the mean squared error (MSE).
      - `print_final_global_model()`: Prints the final global model weights after training.


3. `experiment_runner.py`
   - **Purpose**: Contains the `ExperimentRunner` class, which orchestrates the federated learning experiments by combining dataset generation and federated learning.
   - **Key Functionality**:
      - `run()`: Runs independent experiments, generates training and testing data, executes federated learning, and collects global loss histories for each experiment.


4. `main.py`
   - **Purpose**: The main script that initializes and runs the federated learning experiments using the `ExperimentRunner` class.
   - **Key Functionality**:
      - `main()`: Configures the parameters for the federated learning experiment, invokes the `ExperimentRunner`, and runs the entire process.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Parth-nXp/Online-Federated-Learning.git
    cd Online-Fed
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the main script to start the simulation:
```bash
python main.py
```

## Troubleshooting

If you encounter any issues or errors while running the project, please check the following:

- Ensure all dependencies are installed correctly by running `pip install -r requirements.txt`.
  
- Make sure you are using a compatible version of Python (e.g., Python 3.6 or higher).
 
- If you encounter issues related to missing files or incorrect paths, verify that you are in the correct directory (`Online-Fed`).

If problems persist, feel free to open an issue on GitHub.

## Contributing

Contributions are welcome! If you have suggestions for improvements or bug fixes, please follow these steps:

1. Fork the repository.

2. Create a new branch (`git checkout -b feature-branch`).

3. Make your changes and commit them (`git commit -m 'Add some feature'`).

4. Push to the branch (`git push origin feature-branch`).

5. Open a pull request.

Please ensure your code follows the existing style and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.txt) file for details.
