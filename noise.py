"""
noise.py - the three noise models of our quantum network.

1. Depolarizing noise on the entangled (Bell) pair  -> imperfect entanglement
2. Qubit / photon loss                              -> information loss
3. Bit-flip on the classical channel                -> Alice's 2 bits get corrupted

Convention: p = "quality" of the entanglement (p = 1 perfect, p = 0 pure noise).
The Arduino pot gives NOISE in 0..1, so we use p = 1 - noise.
"""
import numpy as np

# ---- Bell state |Phi+> = (|00> + |11>) / sqrt(2) ---------------------------
PHI_PLUS = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
RHO_PHI_PLUS = np.outer(PHI_PLUS, PHI_PLUS.conj())      # 4x4 density matrix


def noisy_bell_pair(p):
    """
    Depolarized Bell pair (Werner state):
        rho = p * |Phi+><Phi+|  +  (1 - p) * I/4
    With probability p the pair is perfect, otherwise it is fully random noise.
    """
    return p * RHO_PHI_PLUS + (1 - p) * np.eye(4) / 4


def qubit_lost(loss_prob, rng):
    """True if the qubit is lost in the channel (transmission fails completely)."""
    return rng.random() < loss_prob


def flip_classical_bits(bits, flip_prob, rng):
    """Each of Alice's 2 classical bits is flipped independently with prob flip_prob."""
    return tuple(b ^ int(rng.random() < flip_prob) for b in bits)
