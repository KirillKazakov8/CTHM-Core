#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CTHM / KTGR Axioms Stand
------------------------

Python stand that runs small numerical "toy experiments"
for the axioms and layers of the Constructive Theory of Holographic Mind (CTHM).

Covered pieces:

Axiom 1: Knowledge = finite-complexity Hamiltonian H_final(W)
Axiom 2: Master equation and dephasing Σ (coherent → classical)
Axiom 3: Bulk ↔ boundary compression via Dirichlet-to-Neumann (DtN) operator
Axiom 4: SK contour / bulk projection gives the same effective boundary action
Axiom 5: Variational learning of W by minimizing a cost functional J
Section 7.1: Energy ↔ probability (Gibbs distribution over states)
Def. 7.5: Confidence as a function of energy gap ΔE
Axiom 6 / 7.2–7.4: Gibbs / MDL layer on “laws” U (E_meta, P(U|D))
Axiom 8: Creative layer, virtual birth–death controlled by ΔJ + α C_complex

All models here are toy models; they serve as numerical sanity checks
and an executable illustration of the formalism, not as physical experiments.

© 2025 Kirill Kazakov. Academic / research use only.
"""

import numpy as np
from numpy.linalg import norm
from scipy.linalg import expm


# ============================================================
#  Utilities: Pauli matrices and Kronecker products
# ============================================================

def pauli_matrices():
    """Return Pauli matrices for a single qubit."""
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return I, X, Y, Z


def kron(*ops):
    """Kronecker product chain."""
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


# ============================================================
#  Axiom 1: Knowledge = H_final(W)
# ============================================================

def build_H_final(W):
    """
    Minimal illustration of Axiom 1:
    H_final(W) = w_ent * (Z⊗Z) + w_phi * 0.5*(X⊗I + I⊗X) + w_rot * (Y⊗Y)

    Here:
      - w_ent  : “cost” of coherence / entanglement
      - w_phi  : phase structure (U(1)-like)
      - w_rot  : discrete rotations / symmetries
    """
    I, X, Y, Z = pauli_matrices()

    H_ent = kron(Z, Z)
    H_phi = 0.5 * (kron(X, I) + kron(I, X))
    H_rot = kron(Y, Y)

    H = (
        W["w_ent"] * H_ent +
        W["w_phi"] * H_phi +
        W["w_rot"] * H_rot
    )

    channels = {
        "H_ent": H_ent,
        "H_phi": H_phi,
        "H_rot": H_rot
    }
    return H, channels


def test_axiom_1_knowledge():
    """
    Axiom 1 sanity check:
    - H_final(W) is a linear combination of a few fixed operators;
    - |W| << dim(H)^2 (compactness);
    - the same W defines full unitary evolution exp(-i H t).
    """
    print("\n--- [0] Axiom 1: Knowledge = H_final(W) ---")

    W = {"w_ent": 1.3, "w_phi": -0.7, "w_rot": 0.4}
    H, channels = build_H_final(W)

    # 1) Linear decomposition H ≈ Σ w_i H_i
    H_recon = (
        W["w_ent"] * channels["H_ent"] +
        W["w_phi"] * channels["H_phi"] +
        W["w_rot"] * channels["H_rot"]
    )
    diff = norm(H - H_recon)
    print(f"  ||H - Σ w_i H_i|| = {diff:.2e} (expected ~0)")

    # 2) Compactness: few parameters vs full matrix
    dim = H.shape[0]
    num_params = len(W)
    matrix_elements = dim * dim
    compression_ratio = matrix_elements / num_params

    print(f"  dim(H) = {dim} -> matrix has {matrix_elements} elements")
    print(f"  |W| = {num_params}, compression ratio ≈ {compression_ratio:.1f}x")

    # 3) Dynamics: W fully determines evolution
    psi0 = np.array([1, 0, 0, 0], dtype=complex)  # |00>
    t = 0.7
    U = expm(-1j * H * t)
    psi_t = U @ psi0
    prob = np.abs(psi_t) ** 2

    print(f"  Example dynamics: P[00,01,10,11](t={t}) = {np.round(prob.real, 3)}")
    print("  (dynamics and probabilities are defined ONLY via W)")

    cond_decomp = diff < 1e-12
    cond_compress = compression_ratio >= 5.0  # simple compactness sanity check

    if cond_decomp and cond_compress:
        print("  ✅ Axiom 1 OK: knowledge is encoded by compact H_final(W).")
        return True
    else:
        print("  ❌ Axiom 1 FAILED.")
        return False


# ============================================================
#  Axiom 2: Master Equation (coherent → classical)
# ============================================================

def test_axiom_2_dynamics():
    """
    Axiom 2 sanity check:
    Lindblad dephasing term suppresses quantum oscillations and yields
    a classical-looking tail in the population dynamics.
    """
    print("\n--- [1] Axiom 2: Master Equation / Dephasing ---")

    H = np.array([[0, 1], [1, 0]], dtype=complex)      # σ_x
    L_op = np.array([[1, 0], [0, -1]], dtype=complex)  # σ_z (dephasing channel)

    rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    dt = 0.05
    steps = 200

    # Scenario 1: purely coherent (no dephasing)
    p_coh = []
    rho_c = rho0.copy()
    for _ in range(steps):
        comm = -1j * (H @ rho_c - rho_c @ H)
        rho_c += comm * dt
        p_coh.append(rho_c[0, 0].real)

    # Scenario 2: with dephasing
    p_dec = []
    rho_d = rho0.copy()
    gamma = 2.0
    for _ in range(steps):
        comm = -1j * (H @ rho_d - rho_d @ H)
        dissip = (
            L_op @ rho_d @ L_op.conj().T -
            0.5 * (L_op.conj().T @ L_op @ rho_d + rho_d @ L_op.conj().T @ L_op)
        )
        rho_d += (comm + gamma * dissip) * dt
        p_dec.append(rho_d[0, 0].real)

    std_coh = np.std(p_coh)
    std_dec = np.std(p_dec[-50:])  # tail of the decohered dynamics

    print(f"  STD coherent (expected > 0.1): {std_coh:.4f}")
    print(f"  STD with dephasing (expected < 0.01): {std_dec:.4f}")

    if std_coh > 0.1 and std_dec < 0.01:
        print("  ✅ Axiom 2 OK: dephasing kills interference and yields a classical tail.")
        return True
    else:
        print("  ❌ Axiom 2 FAILED.")
        return False


# ============================================================
#  Axiom 3: Holography / DtN operator
# ============================================================

def test_axiom_3_holography():
    """
    Axiom 3 sanity check:
    Bulk Laplacian L on a simple chain (B1 -- I1 -- I2 -- B2) can be compressed
    to a boundary DtN operator Λ such that the boundary flux J matches.
    """
    print("\n--- [2] Axiom 3: Holography / DtN ---")

    L = np.array([
        [1, -1, 0, 0],   # B1
        [-1, 2, -1, 0],  # I1
        [0, -1, 2, -1],  # I2
        [0, 0, -1, 1]    # B2
    ], dtype=float)

    b_idx = [0, 3]
    i_idx = [1, 2]

    L_BB = L[np.ix_(b_idx, b_idx)]
    L_BI = L[np.ix_(b_idx, i_idx)]
    L_IB = L[np.ix_(i_idx, b_idx)]
    L_II = L[np.ix_(i_idx, i_idx)]

    Lambda = L_BB - L_BI @ np.linalg.inv(L_II) @ L_IB

    V_b = np.array([1.0, 0.0])  # potential on B1=1, B2=0

    # 1) DtN flux
    J_eff = Lambda @ V_b

    # 2) Bulk flux with explicit interior solve
    V_i = np.linalg.solve(L_II, -L_IB @ V_b)
    V_full = np.zeros(4)
    V_full[b_idx] = V_b
    V_full[i_idx] = V_i
    J_full = (L @ V_full)[b_idx]

    diff = norm(J_eff - J_full)
    print(f"  ||J_eff - J_full|| = {diff:.2e} (expected ~0)")

    if diff < 1e-10:
        print("  ✅ Axiom 3 OK: bulk compressed into DtN operator on the boundary.")
        return True
    else:
        print("  ❌ Axiom 3 FAILED.")
        return False


# ============================================================
#  Axiom 4: SK contour / bulk projection
# ============================================================

def test_axiom_4_sk_projection():
    """
    Axiom 4 sanity check:
    SK action on the full bulk equals the effective boundary SK action
    after integrating out interior nodes (both coherent and noise parts).

    We explicitly compute:
        S_full(X_+, X_-)   on bulk
        S_eff(X_+^B, X_-^B) on boundary
    and compare them.
    """
    print("\n--- [3] Axiom 4: SK contour / bulk projection ---")

    # Minimal bulk: three nodes (0 and 2 are boundary, 1 is interior)
    L = np.array([
        [1, -1, 0],
        [-1, 2, -1],
        [0, -1, 1]
    ], dtype=float)

    b_idx = [0, 2]
    i_idx = [1]

    L_BB = L[np.ix_(b_idx, b_idx)]
    L_BI = L[np.ix_(b_idx, i_idx)]
    L_IB = L[np.ix_(i_idx, b_idx)]
    L_II = L[np.ix_(i_idx, i_idx)]

    # DtN operator for the coherent part
    Lambda = L_BB - L_BI @ np.linalg.inv(L_II) @ L_IB

    # Effective operator for the noise term (diff^T diff)
    # From diff_I = -L_II^{-1} L_IB diff_B we get:
    # diff^T diff = diff_B^T (I + L_IB^T (L_II^{-T} L_II^{-1}) L_IB) diff_B
    G_eff = np.eye(len(b_idx)) + L_IB.T @ np.linalg.inv(L_II.T @ L_II) @ L_IB

    gamma = 0.3
    rng = np.random.default_rng(42)

    errors = []
    for _ in range(50):
        X_plus_B = rng.normal(size=len(b_idx))
        X_minus_B = rng.normal(size=len(b_idx))

        # Interior nodes on both branches (stationarity in the bulk)
        X_plus_I = np.linalg.solve(L_II, -L_IB @ X_plus_B)
        X_minus_I = np.linalg.solve(L_II, -L_IB @ X_minus_B)

        X_plus = np.zeros(3)
        X_minus = np.zeros(3)
        X_plus[b_idx] = X_plus_B
        X_plus[i_idx] = X_plus_I
        X_minus[b_idx] = X_minus_B
        X_minus[i_idx] = X_minus_I

        diff_full = X_plus - X_minus
        diff_b = X_plus_B - X_minus_B

        # Full SK action (bulk + noise everywhere)
        S_full = (X_plus.T @ L @ X_plus) - (X_minus.T @ L @ X_minus) \
                 + 1j * gamma * (diff_full @ diff_full)

        # Effective SK action on the boundary (DtN + projected noise)
        S_eff = (X_plus_B.T @ Lambda @ X_plus_B) - (X_minus_B.T @ Lambda @ X_minus_B) \
                + 1j * gamma * (diff_b.T @ G_eff @ diff_b)

        errors.append(abs(S_full - S_eff))

    max_err = np.max(errors)
    print(f"  max |S_full - S_eff| = {max_err:.2e}")

    if max_err < 1e-10:
        print("  ✅ Axiom 4 OK: bulk SK action matches the boundary DtN SK action.")
        return True
    else:
        print("  ❌ Axiom 4 FAILED.")
        return False


# ============================================================
#  Axiom 5: Variational learning of W
# ============================================================

def test_axiom_5_learning():
    """
    Axiom 5 sanity check:
    Simplified cost J(W) = (P(W) - target)^2 with P(W) = cos^2(W).
    Gradient descent on W should converge to ~0 for target=1.
    """
    print("\n--- [4] Axiom 5: Variational Learning of W ---")

    target = 1.0
    w = 0.5
    lr = 0.5
    print(f"  Start W: {w:.2f}")

    for _ in range(15):
        prob = (np.cos(w)) ** 2
        loss = (prob - target) ** 2
        # dP/dw = -sin(2w), dL/dw = 2(P - T)*dP/dw
        grad = 2 * (prob - target) * (-np.sin(2 * w))
        w = w - lr * grad

    print(f"  End W: {w:.2f}, Loss: {loss:.6f}")

    if loss < 0.01:
        print("  ✅ Axiom 5 OK: W is updated by minimizing J.")
        return True
    else:
        print("  ❌ Axiom 5 FAILED.")
        return False


# ============================================================
#  Section 7.1: Energy ↔ Probability (Gibbs over states)
# ============================================================

def test_section_7_gibbs_states():
    """
    Section 7.1 sanity check:
    Overdamped Langevin dynamics with potential E(x)=x^2
    converges to the Gibbs distribution P(x) ∝ exp(-β E(x)),
    with Var(x) = 1 / (2β).
    """
    print("\n--- [5] Section 7.1: Energy ↔ Probability (Gibbs over states) ---")

    beta = 2.0
    dt = 0.01
    n_steps = 200
    n_particles = 5000

    particles = np.random.uniform(-3, 3, n_particles)

    for _ in range(n_steps):
        force = -2 * particles  # -dE/dx
        noise = np.random.normal(0, 1, size=n_particles)
        particles += force * dt + np.sqrt(2 * dt / beta) * noise

    var_theory = 1 / (2 * beta)
    var_exp = np.var(particles)

    print(f"  Var theory:     {var_theory:.4f}")
    print(f"  Var experiment: {var_exp:.4f}")

    if abs(var_theory - var_exp) < 0.05:
        print("  ✅ Section 7.1 OK: Langevin dynamics yields Gibbs distribution.")
        return True
    else:
        print("  ❌ Section 7.1 FAILED.")
        return False


# ============================================================
#  Def. 7.5: Confidence as a function of energy gap ΔE
# ============================================================

def test_section_7_confidence_gap():
    """
    Definition 7.5 sanity check:
    conf = 1 - exp(-β_eff * ΔE), where
      ΔE = E_(2) - E_(1)
    should behave consistently with Gibbs probabilities:
      - larger ΔE → larger confidence;
      - conf in [0,1];
      - larger ΔE correlates with larger P_best - P_second.
    """
    print("\n--- [6] Section 7.5: Confidence as Energy Gap ΔE ---")

    def gibbs_prob(energies, beta):
        E = np.array(energies)
        w = np.exp(-beta * E)
        w /= np.sum(w)
        return w

    beta_eff = 5.0

    # Scenario A: small energy gap between best and second-best
    E_A = [0.0, 0.1, 0.2]
    P_A = gibbs_prob(E_A, beta_eff)
    idx_sorted_A = np.argsort(E_A)
    E1_A = E_A[idx_sorted_A[0]]
    E2_A = E_A[idx_sorted_A[1]]
    deltaE_A = E2_A - E1_A
    conf_A = 1.0 - np.exp(-beta_eff * deltaE_A)
    P_best_A = P_A[idx_sorted_A[0]]
    P_second_A = P_A[idx_sorted_A[1]]
    gapP_A = P_best_A - P_second_A

    # Scenario B: large energy gap
    E_B = [0.0, 2.0, 3.0]
    P_B = gibbs_prob(E_B, beta_eff)
    idx_sorted_B = np.argsort(E_B)
    E1_B = E_B[idx_sorted_B[0]]
    E2_B = E_B[idx_sorted_B[1]]
    deltaE_B = E2_B - E1_B
    conf_B = 1.0 - np.exp(-beta_eff * deltaE_B)
    P_best_B = P_B[idx_sorted_B[0]]
    P_second_B = P_B[idx_sorted_B[1]]
    gapP_B = P_best_B - P_second_B

    print(f"  Scenario A (small ΔE):")
    print(f"    Energies: {E_A}")
    print(f"    ΔE_A = {deltaE_A:.3f}, conf_A = {conf_A:.3f}")
    print(f"    P_best - P_second = {gapP_A:.3f}")
    print(f"  Scenario B (large ΔE):")
    print(f"    Energies: {E_B}")
    print(f"    ΔE_B = {deltaE_B:.3f}, conf_B = {conf_B:.3f}")
    print(f"    P_best - P_second = {gapP_B:.3f}")

    conditions = [
        0.0 <= conf_A <= 1.0,
        0.0 <= conf_B <= 1.0,
        conf_B > conf_A + 0.3,         # confidence increases with gap
        gapP_B > gapP_A + 0.1          # probability gap also increases
    ]

    if all(conditions):
        print("  ✅ Def. 7.5 OK: confidence behaves as a monotone function of ΔE.")
        return True
    else:
        print("  ❌ Def. 7.5 FAILED.")
        return False


# ============================================================
#  Axiom 6: Gibbs / MDL layer on “laws” (U)
# ============================================================

def test_axiom_6_prob_layer():
    """
    Axiom 6 sanity check (Sections 7.2–7.4):

    1) For different β, Langevin dynamics on E(x)=x^2 gives different variances
       consistent with Var = 1/(2β); higher temperature (lower β) → broader distribution.
    2) For a small family of models U with different meta-energies E_meta(U)
       (MSE + λ * complexity), the empirical frequency of choices matches
       P(U|D) ∝ exp(-E_meta(U)).
    """
    print("\n--- [7] Axiom 6: Probabilistic / MDL Layer (Hypotheses and Laws) ---")

    def langevin_samples(beta, steps=300, particles=4000, dt=0.01):
        pts = np.random.uniform(-3, 3, particles)
        for _ in range(steps):
            force = -2 * pts
            noise = np.random.normal(0, 1, size=len(pts))
            pts += force * dt + np.sqrt(2 * dt / beta) * noise
        return pts

    betas = [0.5, 2.0]
    var_errors = []
    for beta in betas:
        samples = langevin_samples(beta)
        var_exp = np.var(samples)
        var_theory = 1 / (2 * beta)
        var_errors.append(abs(var_exp - var_theory))
        print(f"  β={beta:.2f}: Var_theory={var_theory:.4f}, Var_exp={var_exp:.4f}")

    # Higher temperature (lower β) should give larger variance
    beta_order_ok = np.var(langevin_samples(betas[0])) > np.var(langevin_samples(betas[1]))

    rng = np.random.default_rng(123)
    x_data = np.linspace(-1, 1, 200)
    y_true = np.sin(np.pi * x_data)
    y_obs = y_true + 0.05 * rng.normal(size=len(x_data))

    models = [
        {"name": "Linear", "deg": 1},
        {"name": "Cubic", "deg": 3},
        {"name": "Quintic", "deg": 5},
    ]

    lambda_complex = 0.01
    E_meta = []
    for m in models:
        coeffs = np.polyfit(x_data, y_obs, deg=m["deg"])
        preds = np.polyval(coeffs, x_data)
        mse = np.mean((preds - y_obs) ** 2)
        complexity = (m["deg"] + 1)
        energy = mse + lambda_complex * complexity
        E_meta.append(energy)
        print(f"  {m['name']}: MSE={mse:.4f}, Complexity={complexity}, E_meta={energy:.4f}")

    beta_meta = 25.0
    E_meta = np.array(E_meta)
    P_theory = np.exp(-beta_meta * E_meta)
    P_theory /= np.sum(P_theory)

    counts = np.zeros(len(models), dtype=int)
    trials = 5000
    for _ in range(trials):
        noise = rng.gumbel(size=len(models))
        scores = -beta_meta * E_meta + noise
        choice = np.argmax(scores)
        counts[choice] += 1
    P_emp = counts / trials

    print("  Theoretical P(U|D):", np.round(P_theory, 3))
    print("  Empirical  P(U|D):", np.round(P_emp, 3))

    var_check = all(err < 0.05 for err in var_errors) and beta_order_ok
    meta_check = np.max(np.abs(P_theory - P_emp)) < 0.05 and np.argmax(P_theory) == np.argmin(E_meta)

    if var_check and meta_check:
        print("  ✅ Axiom 6 OK: hypotheses and laws follow Gibbs/MDL behaviour.")
        return True
    else:
        print("  ❌ Axiom 6 FAILED.")
        return False


# ============================================================
#  Axiom 8: Creative layer / virtual birth–death
# ============================================================

def test_axiom_8_creativity():
    """
    Axiom 8 sanity check:

    Virtual objects are born and die according to
      ΔJ(K) + α C_complex(K) < -ε
    Only those with sufficient gain survive in the long run,
    and the overall cost J should decrease.
    """
    print("\n--- [8] Axiom 8: Creative Layer / Virtual Birth–Death ---")

    base_loss = 1.0
    virtual_catalog = {
        "x^2": 0.25,        # strong useful feature
        "sin(2x)": 0.40,    # even stronger
        "noise_knot": 0.05, # almost useless
        "spurious": 0.08,   # garbage
    }
    alpha = 0.1
    eps = 0.02
    rng = np.random.default_rng(7)

    active = set()
    accepted, removed = 0, 0

    def current_loss(active_set):
        total_gain = sum(virtual_catalog[v] for v in active_set)
        return base_loss - total_gain + alpha * len(active_set)

    iterations = 200
    for _ in range(iterations):
        # B_virt: birth attempt
        if rng.random() < 0.6 and len(active) < len(virtual_catalog):
            candidate = rng.choice(list(set(virtual_catalog.keys()) - active))
            observed_gain = virtual_catalog[candidate] + rng.normal(scale=0.02)
            delta_J = -observed_gain + alpha  # ΔJ + α C_complex
            if delta_J < -eps:
                active.add(candidate)
                accepted += 1
        # D_virt: death / pruning
        elif active:
            candidate = rng.choice(list(active))
            observed_gain = virtual_catalog[candidate] + rng.normal(scale=0.05)
            if (-observed_gain + alpha) > eps:
                active.remove(candidate)
                removed += 1

    initial_loss = base_loss
    final_loss = current_loss(active)

    print(f"  Accepted virtuals: {accepted}, removed: {removed}")
    print(f"  Active virtuals: {sorted(active)}")
    print(f"  J_initial = {initial_loss:.3f}, J_final = {final_loss:.3f}")

    only_useful = all(virtual_catalog[v] > (alpha + eps) for v in active)
    improved = final_loss < initial_loss - 0.15

    if only_useful and improved:
        print("  ✅ Axiom 8 OK: only self-paying virtual objects survive.")
        return True
    else:
        print("  ❌ Axiom 8 FAILED.")
        return False


# ============================================================
#  Runner
# ============================================================

def run_all_tests():
    results = []
    results.append(test_axiom_1_knowledge())
    results.append(test_axiom_2_dynamics())
    results.append(test_axiom_3_holography())
    results.append(test_axiom_4_sk_projection())
    results.append(test_axiom_5_learning())
    results.append(test_section_7_gibbs_states())
    results.append(test_section_7_confidence_gap())
    results.append(test_axiom_6_prob_layer())
    results.append(test_axiom_8_creativity())

    if all(results):
        print("\n=== RESULT: ALL AXIOMS / LAYERS PASSED NUMERICAL SANITY CHECKS. ✅ ===")
    else:
        print("\n=== RESULT: SOME AXIOMS / LAYERS FAILED. ❌ ===")


if __name__ == "__main__":
    run_all_tests()
