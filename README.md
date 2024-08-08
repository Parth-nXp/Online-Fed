# Online Federated Learning with Communication Efficiency

This project implements online federated learning strategies for resource-constrained devices using random Fourier features (RFFs) for kernel regression. The model training is performed in real-time with continuous data streams, ensuring efficient local model updates and maintaining model performance.

This simulation is inspired by the work of Anthony Kuh, titled "Real-Time Kernel Learning for Sensor Networks using Principles of Federated Learning," presented at the APSIPA Annual Summit and Conference 2021 in Tokyo, Japan. [Read the paper](https://ieeexplore.ieee.org/document/9689337).

## Project Structure

The project is divided into three main scripts:

1. **client.py**
   - Contains the `Client` class, which generates synthetic data and calculates random Fourier features.

2. **federated_learning.py**
   - Contains the `federated_learning` function, which performs federated learning across multiple clients.

3. **main.py**
   - The main script that runs the simulation, collects MSE values, and plots the results.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Online-Fed.git
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
