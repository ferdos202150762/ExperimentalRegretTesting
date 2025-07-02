# Experimental Regret Testing

This project explores algorithms for regret testing in normal form games. The primary focus is on implementing and simulating algorithms that converge to Nash equilibria.

## Author

* **Francisco Aristi**
* **License**: MIT

## Project Structure

The project is divided into two main parts:

*   **`numpy/`**: Contains the core implementation using NumPy for numerical operations.
*   **`jax/`**: Contains an alternative implementation using JAX for potential performance improvements and automatic differentiation.

## Getting Started

### Prerequisites

*   Python 3.x
*   NumPy
*   tqdm

You can install the required packages using pip:

```bash
pip install numpy tqdm
```

### Running the Simulations (NumPy)

The `numpy` directory contains several scripts to run simulations. The main simulation is `ExperimentRegretTesting.py`.

To run the default simulation, navigate to the `src` directory and execute the following command:

```bash
python -m numpy.ExperimentRegretTesting 
```

This will run a simulation of a 3-player game and print the results, including the agents' mixed strategies, expected payoffs, and the final regret.

You can also run other experiments, such as:

*   `AnnealedLocExperimentRegretTesting.py`: An experiment with an annealing learning rate.
*   `MatchingPennies3P.py`: A specific example of a 3-player Matching Pennies game.

To run these, use the same format as above:

```bash
python -m numpy.AnnealedLocExperimentRegretTesting
python -m numpy.MatchingPennies3P
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
