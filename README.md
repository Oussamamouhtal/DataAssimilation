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

To run the 4D-Var data assimilation process, you need to have the necessary model, operators, and solver files in place. The main function is fourDvar, which returns a list of values of the quadratic cost function. You can run logger.py to generate a plot showing the convergence of the Gauss-Newton process or plot just the second inner loop to benchmark different inner solvers.


