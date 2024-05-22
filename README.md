# 4D-Var Data Assimilation

This project implements the 4D-Var (Four-Dimensional Variational Data Assimilation) technique to estimate the state of a dynamic system. The code is designed to work with the Lorenz '95 model, a simplified atmospheric model often used in data assimilation studies.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Functions](#functions)
- [References](#references)

## Installation

To set up the environment, follow these steps:

1. **Clone the repository:**
    ```sh
    git clone https://github.com/Oussamamouhtal/DataAssimilation.git
    cd DataAssimilation
    ```

2. **Create a virtual environment:**
    ```sh
    python -m venv env
    ```

3. **Activate the virtual environment:**
    - On Windows:
        ```sh
        .\env\Scripts\activate
        ```
    - On macOS and Linux:
        ```sh
        source env/bin/activate
        ```

4. **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To run the 4D-Var data assimilation process, you need to have the necessary model, operators, and solver files in place. The main function is `fourDvar`, which drives the data assimilation process.

### Example Usage

Below is an example script to run the 4D-Var process:

```python
from fourdvar import fourDvar

# Parameters
n = 120  # state space dimension
m_t = 30  # number of observations at a fixed time
Nt = 2  # total number of observations at a fixed location
max_outer = 2  # max number of outer loops (Gauss-Newton loop)
max_inner = 40  # max number of inner loops
method = "Spectral_LMP"  # Solving inner loops with 
                        # method with iterative methode 
selectedTHETA = "mediane"
IP = False  # Start the solver with an appropriate initial guess.

# Run 4D-Var
fourDvar(n, m_t, Nt, max_outer, max_inner, method, selectedTHETA, IP )
