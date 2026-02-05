# Quantum Memory Inference and Unitary Construction
Python implementation for inferring quantum model from a binary time series and constructing the corresponding unitary operator for quantum simulation.

The implementation follows the frameworks introduced in:
- Quantum memory inference: Phys. Rev. A 101, 032327 (2020)  
- Unitary construction for quantum models: Phys. Rev. Lett. 120, 240502 (2018)

This code is intended for research use and numerical experimentation.

---

## Features

- Infers quantum memory states from binary stochastic processes  
- Merges equivalent states using an overlap-based equivalence relation  
- Automatically adjusts tolerance to enforce unifilarity  
- Constructs the unitary operator implementing the quantum model  
- Computes the quantum statistical memory (Cq)  

---

## Requirements

Python 3.9+

Dependencies:
numpy

## Usage

Basic example:

```python
import numpy as np
from compute_cq_build_unitary import compute_cq_and_unitary

# Generate random binary time series
bits = np.random.randint(0, 2, 10000)

# Compute quantum statistical memory, unitary, and memory states
Cq, Unitary, QMS = compute_cq_and_unitary(bits, delta=0.001, L=2)

print("Cq:", Cq)
print("Unitary shape:", Unitary.shape)
print("Quantum memory states:\n", QMS)
```

Alternate example:

```python
import numpy as np
from compute_cq_build_unitary import compute_cq_and_unitary

# Generate a time series from the perturbed coin process
def perturbed_coin(p=0.2, bitstring_length=int(1e3), seed=42):
    """
    Generate a binary time series from a two-state perturbed coin process.

    The process is a Markov order 1 chain with two internal states. 
    With probability p, the internal state flips; otherwise it remains the same.
    The emitted symbol depends on the current state, producing correlated
    binary output.

    Arguments:
        p (float, optional): Probability of switching internal state at each
            timestep (default is 0.2).
        bitstring_length (int, optional): Length of the generated binary time
            series (default is 1000).
        seed (int, optional): Random number generator seed (default is 42).
    Returns:
        bits (np.ndarray): 1D numpy array of {0, 1} representing the generated
            binary time series.
    """

    
    rng = np.random.default_rng(seed)
    state = np.random.randint(2)+1
    bits = np.empty(bitstring_length, dtype=int)
    for ii in range(bitstring_length):
        flip = rng.random() < p
            
        if state == 1:
            if flip:
                state = 2
                bits[ii] = 1
            else:
                bits[ii] = 0
        else:
            if flip:
                state = 1
                bits[ii] = 0    
            else:
                bits[ii] = 1
                
    return np.array(bits)

bits = perturbed_coin(bitstring_length=int(1e4))

# Compute quantum statistical memory, unitary, and memory states
Cq, Unitary, QMS = compute_cq_and_unitary(bits, delta=0.001, L=2)

print("Cq:", Cq)
print("Unitary shape:", Unitary.shape)
print("Quantum memory states:\n", QMS)
```
