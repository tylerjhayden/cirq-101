# ABOUTME: Section 1.5 - Noisy Simulation: noise channels, density matrices, and comparison with ideal simulation

import cirq
import numpy as np
import matplotlib.pyplot as plt


def _print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(title)
    print("="*60)


def demonstrate_noise_channels():
    """Demonstrate different types of noise channels in Cirq."""
    _print_section_header("NOISE CHANNELS")

    q = cirq.LineQubit(0)

    print("\nNoise channels model realistic quantum errors:")
    print("\n1. Bit Flip Channel - random X errors")
    print("   Physical interpretation: Qubit spontaneously flips")
    bit_flip = cirq.bit_flip(p=0.2)
    print(f"   Channel: {bit_flip}")
    print(f"   Operation on qubit: {bit_flip(q)}")

    print("\n2. Depolarizing Channel - symmetric white noise")
    print("   Physical interpretation: Qubit randomly subjected to X, Y, or Z")
    depolarize = cirq.depolarize(p=0.1)
    print(f"   Channel: {depolarize}")
    print(f"   Operation on qubit: {depolarize(q)}")

    print("\n3. Amplitude Damping - energy relaxation (T1 decay)")
    print("   Physical interpretation: Qubit loses energy to environment")
    amp_damp = cirq.amplitude_damp(gamma=0.3)
    print(f"   Channel: {amp_damp}")
    print(f"   Operation on qubit: {amp_damp(q)}")

    print("\n4. Phase Damping - phase relaxation (T2 decay)")
    print("   Physical interpretation: Qubit loses phase coherence")
    phase_damp = cirq.phase_damp(gamma=0.25)
    print(f"   Channel: {phase_damp}")
    print(f"   Operation on qubit: {phase_damp(q)}")


def demonstrate_kraus_operators():
    """Demonstrate Kraus operator representation of noise channels."""
    _print_section_header("KRAUS OPERATOR REPRESENTATION")

    print("\nNoise channels are described by Kraus operators {A_k}:")
    print("ρ → Σ_k A_k ρ A_k†")
    print("\nCompleteness relation: Σ_k A_k† A_k = I")

    # Bit flip example
    print("\n1. Bit Flip Channel (p=0.2):")
    bit_flip = cirq.bit_flip(0.2)
    kraus_ops = cirq.kraus(bit_flip)

    print(f"   Number of Kraus operators: {len(kraus_ops)}")
    for i, K in enumerate(kraus_ops):
        print(f"\n   A_{i}:")
        print(f"   {np.round(K, 3)}")

    # Verify completeness
    sum_kraus = sum(K.conj().T @ K for K in kraus_ops)
    print(f"\n   Completeness check (Σ A_k† A_k):")
    print(f"   {np.round(sum_kraus, 6)}")
    print(f"   Is identity? {np.allclose(sum_kraus, np.eye(2))}")

    # Amplitude damping example
    print("\n2. Amplitude Damping Channel (γ=0.3):")
    amp_damp = cirq.amplitude_damp(0.3)
    kraus_ops = cirq.kraus(amp_damp)

    print(f"   Number of Kraus operators: {len(kraus_ops)}")
    for i, K in enumerate(kraus_ops):
        print(f"\n   A_{i}:")
        print(f"   {np.round(K, 3)}")


def demonstrate_density_matrix_simulation():
    """Demonstrate density matrix simulation for noisy quantum circuits."""
    _print_section_header("DENSITY MATRIX SIMULATION")

    print("\nDensity matrices describe both pure and mixed quantum states")
    print("Pure state: ρ = |ψ⟩⟨ψ|")
    print("Mixed state: ρ = Σ_i p_i |ψ_i⟩⟨ψ_i|")

    q = cirq.LineQubit(0)

    # Pure state example
    print("\n1. Pure State (|+⟩ superposition):")
    pure_circuit = cirq.Circuit(cirq.H(q))

    sv_simulator = cirq.Simulator()
    sv_result = sv_simulator.simulate(pure_circuit)
    psi = sv_result.final_state_vector

    dm_simulator = cirq.DensityMatrixSimulator()
    dm_result = dm_simulator.simulate(pure_circuit)
    rho_pure = dm_result.final_density_matrix

    print(f"\n   State vector |ψ⟩:")
    print(f"   {np.round(psi, 3)}")
    print(f"\n   Density matrix ρ = |ψ⟩⟨ψ|:")
    print(f"   {np.round(rho_pure, 3)}")

    # Calculate purity
    purity_pure = np.trace(rho_pure @ rho_pure).real
    print(f"\n   Purity Tr(ρ²) = {purity_pure:.6f}")
    print(f"   Pure state has purity = 1")

    # Mixed state example
    print("\n2. Mixed State (after depolarizing noise):")
    mixed_circuit = cirq.Circuit(
        cirq.H(q),
        cirq.depolarize(0.3)(q)
    )

    dm_result = dm_simulator.simulate(mixed_circuit)
    rho_mixed = dm_result.final_density_matrix

    print(f"\n   Density matrix (with noise):")
    print(f"   {np.round(rho_mixed, 3)}")

    purity_mixed = np.trace(rho_mixed @ rho_mixed).real
    print(f"\n   Purity Tr(ρ²) = {purity_mixed:.6f}")
    print(f"   Mixed state has purity < 1")

    # Verify density matrix properties
    print("\n3. Density Matrix Properties:")
    print(f"   • Hermitian: ρ = ρ†? {np.allclose(rho_mixed, rho_mixed.conj().T)}")
    print(f"   • Unit trace: Tr(ρ) = 1? {np.allclose(np.trace(rho_mixed), 1.0)}")
    eigenvalues = np.linalg.eigvalsh(rho_mixed)
    print(f"   • Positive semi-definite (eigenvalues ≥ 0)? {np.all(eigenvalues >= -1e-10)}")
    print(f"   • Eigenvalues: {np.round(eigenvalues, 6)}")


def compare_ideal_vs_noisy():
    """Compare ideal and noisy simulation of Bell state."""
    _print_section_header("IDEAL VS NOISY SIMULATION")

    q0, q1 = cirq.LineQubit.range(2)

    print("\nComparing Bell state preparation with and without noise:")

    # Ideal Bell state
    print("\n1. IDEAL Simulation:")
    ideal_circuit = cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0, q1)
    )

    print(f"\nCircuit:")
    print(ideal_circuit)

    dm_sim = cirq.DensityMatrixSimulator()
    ideal_result = dm_sim.simulate(ideal_circuit)
    rho_ideal = ideal_result.final_density_matrix

    print(f"\nDensity matrix:")
    print(np.round(rho_ideal, 3))

    purity_ideal = np.trace(rho_ideal @ rho_ideal).real
    print(f"\nPurity: {purity_ideal:.6f} (pure state)")

    # Noisy Bell state
    print("\n2. NOISY Simulation:")
    noisy_circuit = cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0, q1),
        cirq.depolarize(0.1)(q0),
        cirq.depolarize(0.1)(q1)
    )

    print(f"\nCircuit (with noise):")
    print(noisy_circuit)

    noisy_result = dm_sim.simulate(noisy_circuit)
    rho_noisy = noisy_result.final_density_matrix

    print(f"\nDensity matrix:")
    print(np.round(rho_noisy, 3))

    purity_noisy = np.trace(rho_noisy @ rho_noisy).real
    print(f"\nPurity: {purity_noisy:.6f} (mixed state)")

    print(f"\nPurity reduction: {(1 - purity_noisy/purity_ideal)*100:.1f}%")

    # Measurement statistics
    print("\n3. Measurement Statistics:")

    ideal_circuit_with_measure = ideal_circuit + cirq.Circuit(
        cirq.measure(q0, q1, key='result')
    )
    noisy_circuit_with_measure = noisy_circuit + cirq.Circuit(
        cirq.measure(q0, q1, key='result')
    )

    # Run simulations
    ideal_run = dm_sim.run(ideal_circuit_with_measure, repetitions=1000)
    noisy_run = dm_sim.run(noisy_circuit_with_measure, repetitions=1000)

    ideal_counts = ideal_run.histogram(key='result')
    noisy_counts = noisy_run.histogram(key='result')

    print("\n   Ideal outcomes:")
    for outcome in [0, 1, 2, 3]:
        count = ideal_counts.get(outcome, 0)
        print(f"   |{outcome:02b}⟩: {count:4d} ({count/10:.1f}%)")

    print("\n   Noisy outcomes:")
    for outcome in [0, 1, 2, 3]:
        count = noisy_counts.get(outcome, 0)
        print(f"   |{outcome:02b}⟩: {count:4d} ({count/10:.1f}%)")


def demonstrate_amplitude_damping():
    """Demonstrate amplitude damping (T1 decay) effects."""
    _print_section_header("AMPLITUDE DAMPING (T1 Decay)")

    print("\nAmplitude damping models energy relaxation:")
    print("Excited state |1⟩ decays to ground state |0⟩")

    q = cirq.LineQubit(0)

    # Prepare excited state
    print("\n1. Prepare |1⟩ state (excited):")

    gamma_values = [0.0, 0.2, 0.5, 0.8]
    dm_sim = cirq.DensityMatrixSimulator()

    print("\n2. Apply amplitude damping with varying γ:")
    print(f"\n   γ      |0⟩ pop   |1⟩ pop   Purity")
    print(f"   {'─'*40}")

    populations = []
    for gamma in gamma_values:
        circuit = cirq.Circuit(
            cirq.X(q),
            cirq.amplitude_damp(gamma)(q)
        )

        result = dm_sim.simulate(circuit)
        rho = result.final_density_matrix

        pop_0 = rho[0, 0].real
        pop_1 = rho[1, 1].real
        purity = np.trace(rho @ rho).real

        populations.append((pop_0, pop_1))

        print(f"   {gamma:.1f}    {pop_0:.3f}    {pop_1:.3f}    {purity:.3f}")

    print("\n   As γ increases:")
    print("   • |0⟩ population increases (energy loss)")
    print("   • |1⟩ population decreases")
    print("   • State becomes mixed (purity < 1)")


def demonstrate_phase_damping():
    """Demonstrate phase damping (T2 decay) effects."""
    _print_section_header("PHASE DAMPING (T2 Decay)")

    print("\nPhase damping models loss of phase coherence:")
    print("Superposition states lose off-diagonal density matrix elements")

    q = cirq.LineQubit(0)

    # Prepare superposition
    print("\n1. Prepare |+⟩ = (|0⟩ + |1⟩)/√2:")

    gamma_values = [0.0, 0.2, 0.5, 0.8]
    dm_sim = cirq.DensityMatrixSimulator()

    print("\n2. Apply phase damping with varying γ:")
    print(f"\n   γ      ρ₀₀     ρ₁₁     |ρ₀₁|    Purity")
    print(f"   {'─'*50}")

    for gamma in gamma_values:
        circuit = cirq.Circuit(
            cirq.H(q),
            cirq.phase_damp(gamma)(q)
        )

        result = dm_sim.simulate(circuit)
        rho = result.final_density_matrix

        rho_00 = rho[0, 0].real
        rho_11 = rho[1, 1].real
        rho_01 = abs(rho[0, 1])
        purity = np.trace(rho @ rho).real

        print(f"   {gamma:.1f}    {rho_00:.3f}   {rho_11:.3f}   {rho_01:.3f}   {purity:.3f}")

    print("\n   As γ increases:")
    print("   • Diagonal elements (populations) stay constant")
    print("   • Off-diagonal elements (coherences) decay to zero")
    print("   • State becomes classically mixed")


def visualize_noise_effects():
    """Visualize the effects of different noise channels."""
    _print_section_header("NOISE CHANNEL VISUALIZATION")

    q = cirq.LineQubit(0)

    # Prepare superposition state
    base_circuit = cirq.Circuit(cirq.H(q))

    noise_channels = [
        ("Bit Flip", cirq.bit_flip(0.3)),
        ("Depolarize", cirq.depolarize(0.3)),
        ("Amplitude Damp", cirq.amplitude_damp(0.3)),
        ("Phase Damp", cirq.phase_damp(0.3))
    ]

    dm_sim = cirq.DensityMatrixSimulator()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Ideal state
    ideal_result = dm_sim.simulate(base_circuit)
    rho_ideal = ideal_result.final_density_matrix

    # Plot ideal density matrix (real and imaginary parts)
    im0 = axes[0].imshow(np.real(rho_ideal), cmap='RdBu', vmin=-1, vmax=1)
    axes[0].set_title("Ideal State\n(Real part)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Row')
    plt.colorbar(im0, ax=axes[0])

    axes[1].imshow(np.imag(rho_ideal), cmap='RdBu', vmin=-1, vmax=1)
    axes[1].set_title("Ideal State\n(Imaginary part)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Column')
    axes[1].set_ylabel('Row')
    plt.colorbar(im0, ax=axes[1])

    # Plot each noise channel
    for idx, (name, channel) in enumerate(noise_channels):
        circuit = base_circuit + cirq.Circuit(channel(q))
        result = dm_sim.simulate(circuit)
        rho = result.final_density_matrix

        # Plot real part
        ax_idx = idx + 2
        im = axes[ax_idx].imshow(np.real(rho), cmap='RdBu', vmin=-1, vmax=1)
        purity = np.trace(rho @ rho).real
        axes[ax_idx].set_title(f"{name}\nPurity: {purity:.3f}",
                               fontsize=12, fontweight='bold')
        axes[ax_idx].set_xlabel('Column')
        axes[ax_idx].set_ylabel('Row')
        plt.colorbar(im, ax=axes[ax_idx])

    plt.suptitle("Density Matrix Real Parts: Ideal vs Noisy States",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    print("\nDisplaying density matrix visualization...")
    plt.savefig('notebooks/noisy_simulation_density_matrices.png', dpi=150, bbox_inches='tight')
    print("  Saved to: notebooks/noisy_simulation_density_matrices.png")
    plt.show()


def visualize_purity_vs_noise():
    """Visualize how purity decreases with increasing noise strength."""
    _print_section_header("PURITY VS NOISE STRENGTH")

    q = cirq.LineQubit(0)
    base_circuit = cirq.Circuit(cirq.H(q))

    # Vary noise parameter
    noise_params = np.linspace(0, 0.99, 50)

    channels = {
        'Bit Flip': lambda p: cirq.bit_flip(p),
        'Depolarize': lambda p: cirq.depolarize(p),
        'Amplitude Damp': lambda p: cirq.amplitude_damp(p),
        'Phase Damp': lambda p: cirq.phase_damp(p)
    }

    dm_sim = cirq.DensityMatrixSimulator()

    plt.figure(figsize=(10, 6))

    for name, channel_func in channels.items():
        purities = []

        for p in noise_params:
            circuit = base_circuit + cirq.Circuit(channel_func(p)(q))
            result = dm_sim.simulate(circuit)
            rho = result.final_density_matrix
            purity = np.trace(rho @ rho).real
            purities.append(purity)

        plt.plot(noise_params, purities, label=name, linewidth=2, marker='o',
                markersize=3, markevery=5)

    plt.xlabel('Noise Parameter (p or γ)', fontsize=12)
    plt.ylabel('Purity Tr(ρ²)', fontsize=12)
    plt.title('Purity Degradation with Increasing Noise', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)

    # Add reference line for pure state
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.3, label='Pure state')

    plt.tight_layout()

    print("\nDisplaying purity vs noise strength plot...")
    plt.savefig('notebooks/purity_vs_noise.png', dpi=150, bbox_inches='tight')
    print("  Saved to: notebooks/purity_vs_noise.png")
    plt.show()


def visualize_bell_state_fidelity():
    """Visualize Bell state fidelity degradation under noise."""
    _print_section_header("BELL STATE FIDELITY UNDER NOISE")

    print("\nFidelity F(ρ, σ) = Tr(√(√ρ σ √ρ))² measures state similarity")
    print("Perfect fidelity = 1, completely different = 0")

    q0, q1 = cirq.LineQubit.range(2)

    # Ideal Bell state
    ideal_circuit = cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0, q1)
    )

    dm_sim = cirq.DensityMatrixSimulator()
    ideal_result = dm_sim.simulate(ideal_circuit)
    rho_ideal = ideal_result.final_density_matrix

    # Vary noise strength
    noise_levels = np.linspace(0, 0.5, 30)
    fidelities_depol = []
    fidelities_amp_damp = []

    for noise_p in noise_levels:
        # Depolarizing noise
        noisy_circuit_depol = cirq.Circuit(
            cirq.H(q0),
            cirq.CNOT(q0, q1),
            cirq.depolarize(noise_p)(q0),
            cirq.depolarize(noise_p)(q1)
        )
        result_depol = dm_sim.simulate(noisy_circuit_depol)
        rho_noisy_depol = result_depol.final_density_matrix

        # Fidelity calculation (valid when rho_ideal is pure): F = |Tr(ρ†σ)|
        # For general mixed states, use full formula: F(ρ, σ) = Tr(√(√ρ σ √ρ))²
        fidelity_depol = np.abs(np.trace(rho_ideal.conj().T @ rho_noisy_depol))
        fidelities_depol.append(fidelity_depol)

        # Amplitude damping
        noisy_circuit_amp = cirq.Circuit(
            cirq.H(q0),
            cirq.CNOT(q0, q1),
            cirq.amplitude_damp(noise_p)(q0),
            cirq.amplitude_damp(noise_p)(q1)
        )
        result_amp = dm_sim.simulate(noisy_circuit_amp)
        rho_noisy_amp = result_amp.final_density_matrix

        # Fidelity calculation (valid when rho_ideal is pure)
        fidelity_amp = np.abs(np.trace(rho_ideal.conj().T @ rho_noisy_amp))
        fidelities_amp_damp.append(fidelity_amp)

    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, fidelities_depol, label='Depolarizing Noise',
            linewidth=2, marker='o', markersize=4)
    plt.plot(noise_levels, fidelities_amp_damp, label='Amplitude Damping',
            linewidth=2, marker='s', markersize=4)

    plt.xlabel('Noise Parameter', fontsize=12)
    plt.ylabel('Fidelity with Ideal Bell State', fontsize=12)
    plt.title('Bell State Degradation Under Different Noise Models',
             fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.xlim(0, 0.5)
    plt.ylim(0, 1.05)

    plt.tight_layout()

    print("\nDisplaying Bell state fidelity plot...")
    plt.savefig('notebooks/bell_state_fidelity.png', dpi=150, bbox_inches='tight')
    print("  Saved to: notebooks/bell_state_fidelity.png")
    plt.show()


def main():
    """Run all demonstrations for Section 1.5."""
    print("\n" + "#"*60)
    print("# SECTION 1.5: NOISY SIMULATION")
    print("# Noise Channels, Density Matrices, and Realistic Modeling")
    print("#"*60)

    demonstrate_noise_channels()
    demonstrate_kraus_operators()
    demonstrate_density_matrix_simulation()
    compare_ideal_vs_noisy()
    demonstrate_amplitude_damping()
    demonstrate_phase_damping()
    visualize_noise_effects()
    visualize_purity_vs_noise()
    visualize_bell_state_fidelity()

    print("\n" + "#"*60)
    print("# Section 1.5 Complete!")
    print("# ")
    print("# Key Takeaways:")
    print("# • Noise channels model realistic quantum errors")
    print("# • Density matrices describe both pure and mixed states")
    print("# • Different noise types (bit flip, depolarize, T1, T2)")
    print("# • Noise degrades quantum state purity and fidelity")
    print("# • DensityMatrixSimulator enables noisy simulation")
    print("#"*60 + "\n")


if __name__ == "__main__":
    main()
