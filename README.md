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
from compute_cq import compute_cq_and_unitary

# Generate random binary time series
bits = np.random.randint(0, 2, 10000)

# Compute quantum statistical memory, unitary, and memory states
Cq, Unitary, QMS = compute_cq_and_unitary(bits, delta=0.001, L=2)

print("Cq:", Cq)
print("Unitary shape:", Unitary.shape)
print("Quantum memory states:\n", QMS)
```

