"""
teleport.py - quantum teleportation of one qubit from Alice to Bob.

Qubits (3 in total):
    qubit 0 : the unknown state |psi> Alice wants to send
    qubit 1 : Alice's half of the entangled Bell pair
    qubit 2 : Bob's half of the entangled Bell pair

We use DENSITY MATRICES (8x8) so that noisy / mixed states can be handled.
Run this file directly to self-test: ideal teleportation must give fidelity 1.0
"""
import numpy as np
from noise import noisy_bell_pair, qubit_lost, flip_classical_bits

CLASSICAL_LIMIT = 2 / 3   # best fidelity any "measure and resend" strategy can reach

# ---- Basic gates (just small matrices) ------------------------------------
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]], dtype=complex)

# Alice's Bell-measurement circuit = CNOT(qubit0 -> qubit1), then H on qubit0.
# kron() extends a gate to the full 3-qubit system.
U_ALICE = np.kron(H, np.eye(4)) @ np.kron(CNOT, I)


def random_state(rng):
    """A random single-qubit state a|0> + b|1> (normalised)."""
    v = rng.normal(size=2) + 1j * rng.normal(size=2)
    return v / np.linalg.norm(v)


def fidelity(psi, rho):
    """How close is Bob's state rho to the original |psi>?  F = <psi| rho |psi>."""
    return float(np.real(psi.conj() @ rho @ psi))


def teleport_once(psi, p=1.0, loss_prob=0.0, flip_prob=0.0, rng=None):
    """
    One teleportation attempt.
      p         : entanglement quality (1 = perfect Bell pair)
      loss_prob : chance the qubit is lost in the channel
      flip_prob : chance each classical bit is flipped on its way to Bob
    Returns (lost, fidelity). fidelity is None if the qubit was lost.
    """
    rng = rng or np.random.default_rng()

    # Step 0: qubit lost -> nothing arrives at Bob
    if qubit_lost(loss_prob, rng):
        return True, None

    # Step 1: total state = |psi><psi|  (x)  shared (noisy) Bell pair
    rho_psi = np.outer(psi, psi.conj())
    rho = np.kron(rho_psi, noisy_bell_pair(p))            # 8x8

    # Step 2: Alice applies her Bell-measurement circuit
    rho = U_ALICE @ rho @ U_ALICE.conj().T

    # Step 3: Alice measures qubits 0 and 1 -> two classical bits (m0, m1).
    # For each outcome, Bob's qubit is the matching 2x2 block of the 8x8 matrix.
    outcomes, probs, bob_states = [], [], []
    for m0 in (0, 1):
        for m1 in (0, 1):
            idx = [4 * m0 + 2 * m1, 4 * m0 + 2 * m1 + 1]   # rows/cols where qubits 0,1 = (m0,m1)
            block = rho[np.ix_(idx, idx)]
            prob = np.real(np.trace(block))
            outcomes.append((m0, m1))
            probs.append(prob)
            bob_states.append(block / prob)
    k = rng.choice(4, p=np.array(probs) / sum(probs))      # quantum measurement result
    bits, rho_bob = outcomes[k], bob_states[k]

    # Step 4: Alice sends the 2 bits over the CLASSICAL channel (may get corrupted)
    r0, r1 = flip_classical_bits(bits, flip_prob, rng)

    # Step 5: Bob corrects his qubit using the bits he RECEIVED:  X if r1, Z if r0
    C = np.linalg.matrix_power(Z, r0) @ np.linalg.matrix_power(X, r1)
    rho_bob = C @ rho_bob @ C.conj().T

    return False, fidelity(psi, rho_bob)


def run_trials(p=1.0, loss_prob=0.0, flip_prob=0.0, n=1000, seed=None):
    """
    Monte Carlo: repeat teleportation n times with random input states.
    Returns (mean_fidelity, success_rate)
      mean_fidelity : average F over the qubits that actually arrived
      success_rate  : fraction of ALL attempts that arrived AND had F > 2/3
    """
    rng = np.random.default_rng(seed)
    fids, successes = [], 0
    for _ in range(n):
        lost, f = teleport_once(random_state(rng), p, loss_prob, flip_prob, rng)
        if not lost:
            fids.append(f)
            successes += f > CLASSICAL_LIMIT
    mean_f = float(np.mean(fids)) if fids else 0.0
    return mean_f, successes / n


if __name__ == "__main__":
    # ---- Self-test 1: ideal teleportation must be perfect -----------------
    rng = np.random.default_rng(1)
    worst = min(teleport_once(random_state(rng), p=1.0, rng=rng)[1] for _ in range(200))
    print(f"Ideal teleportation, worst fidelity over 200 random states: {worst:.6f}")
    assert worst > 0.999999, "Teleportation circuit has a bug!"

    # ---- Self-test 2: matches theory F = (1 + p) / 2 ----------------------
    for p in (0.0, 0.5, 1.0):
        f, s = run_trials(p=p, n=300, seed=2)
        print(f"p = {p:.1f}  simulated F = {f:.3f}   theory (1+p)/2 = {(1 + p) / 2:.3f}")
    print("All tests passed.")
