import numpy as np

from compute_cq_build_unitary import compute_cq_and_unitary

def perturbed_coin(p=0.2, q=0.2, bitstring_length=int(1e3), seed=42):
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
        if state == 1:
            if rng.random() < p:
                state = 2
                bits[ii] = 1
            else:
                bits[ii] = 0
        else:
            if rng.random() < q:
                state = 1
                bits[ii] = 0
            else:
                bits[ii] = 1

    return np.array(bits)

def main():
    # Generate bitstring
    bits = perturbed_coin(p=0.2, q=0.2, bitstring_length=int(1e4), seed=42)

    # Run quantum inference protocol
    Cq, Unitary, QMS = compute_cq_and_unitary(bits, delta=int(1e-6), L=2)

    print("Quantum statistical memory =", Cq)
    print("Unitary =", np.around(Unitary, 4))
    print("Unitary shape =", Unitary.shape)
    print("Quantum memory states =", np.around(QMS, 4))


if __name__ == "__main__":
    main()

