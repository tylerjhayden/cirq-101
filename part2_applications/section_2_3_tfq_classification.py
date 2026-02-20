# ABOUTME: Demonstrates hybrid quantum-classical machine learning with parameterized quantum circuits
# ABOUTME: Simplified implementation showing quantum ML concepts without TensorFlow Quantum

import cirq
import sympy
import numpy as np
from typing import List, Tuple, Dict, Any


def _print_major_header(title: str) -> None:
    """
    Print a major section header with equals signs.

    Args:
        title: The header text to print
    """
    print("=" * 70)
    print(title)
    print("=" * 70)


def _print_section_header(title: str) -> None:
    """
    Print a section header with dashes.

    Args:
        title: The header text to print
    """
    print("-" * 70)
    print(title)
    print("-" * 70)


def is_tfq_available() -> bool:
    """
    Check if TensorFlow Quantum is available.

    Returns:
        bool: True if TFQ is installed, False otherwise
    """
    try:
        import tensorflow_quantum
        return True
    except ImportError:
        return False


def get_implementation_notes() -> str:
    """
    Return notes about this implementation.

    Returns:
        str: Implementation notes explaining TFQ status
    """
    if is_tfq_available():
        return "Full TensorFlow Quantum implementation with trainable quantum layers."
    else:
        return """
        SIMPLIFIED IMPLEMENTATION - TensorFlow Quantum Not Available

        This demonstrates quantum machine learning concepts without tensorflow-quantum:
        - Data encoding into quantum states (amplitude and angle encoding)
        - Parameterized quantum circuits (PQC) for feature extraction
        - Measurement strategies for extracting classical outputs
        - Conceptual hybrid quantum-classical architecture

        For full implementation with gradient-based training, install:
            pip install tensorflow-quantum tensorflow>=2.11.0

        The full TFQ implementation would enable:
        - Automatic differentiation through quantum circuits
        - Integration with Keras layers (tfq.layers.PQC)
        - End-to-end training of hybrid models
        - Scalable batch processing
        """


def amplitude_encode_data(qubits: List[cirq.Qid], data: np.ndarray) -> cirq.Circuit:
    """
    Encode classical data into quantum state amplitudes.

    Amplitude encoding maps a normalized classical vector directly into quantum
    state amplitudes. For n qubits, can encode 2^n values.

    Note: This implementation assumes real-valued amplitudes. For complex
    amplitudes, both angle and phase information would be required, which
    would necessitate a more sophisticated state preparation circuit.

    Args:
        qubits: List of qubits to encode data into
        data: Normalized classical data vector (must have norm=1, real-valued)

    Returns:
        Circuit that prepares the encoded state
    """
    # Verify normalization
    norm = np.linalg.norm(data)
    if not np.isclose(norm, 1.0):
        raise ValueError(f"Data must be normalized (norm={norm:.4f})")

    # For this simplified version, we use state preparation
    # In practice, this would use more sophisticated encoding circuits
    circuit = cirq.Circuit()

    # Simple example: encode 2D data into 1 qubit
    # data = [a, b] -> |ψ> = a|0> + b|1>
    if len(qubits) == 1 and len(data) == 2:
        # Calculate angle from amplitudes
        if np.abs(data[0]) > 1e-10:
            theta = 2 * np.arctan2(data[1], data[0])
            circuit.append(cirq.ry(theta)(qubits[0]))
    else:
        # For general case, this would require more complex state preparation
        # Here we provide a placeholder showing the concept
        raise NotImplementedError(
            "Amplitude encoding for >2 amplitudes requires advanced state preparation. "
            "Use angle encoding for multi-qubit cases."
        )

    return circuit


def angle_encode_data(qubits: List[cirq.Qid], features: List[float]) -> cirq.Circuit:
    """
    Encode classical data as rotation angles.

    Angle encoding maps each classical feature to a rotation angle on a qubit.
    This is simple, hardware-efficient, and commonly used in quantum ML.

    Args:
        qubits: List of qubits (one per feature)
        features: Classical feature values

    Returns:
        Circuit encoding the features as rotations
    """
    if len(qubits) < len(features):
        raise ValueError(f"Need at least {len(features)} qubits for {len(features)} features")

    circuit = cirq.Circuit()

    # Encode each feature as a Y rotation
    # RY(θ) prepares cos(θ/2)|0> + sin(θ/2)|1>
    for qubit, feature in zip(qubits, features):
        circuit.append(cirq.ry(feature)(qubit))

    return circuit


def build_pqc(qubits: List[cirq.Qid], num_layers: int = 1) -> Tuple[cirq.Circuit, List[sympy.Symbol]]:
    """
    Build a parameterized quantum circuit for feature extraction.

    The PQC consists of alternating layers of:
    1. Parameterized single-qubit rotations (trainable)
    2. Entangling two-qubit gates (fixed)

    This architecture is expressive while remaining shallow for NISQ devices.

    Args:
        qubits: Qubits to build the circuit on
        num_layers: Number of PQC layers

    Returns:
        Tuple of (circuit, list of symbolic parameters)
    """
    circuit = cirq.Circuit()
    params = []

    n_qubits = len(qubits)
    param_idx = 0

    for layer in range(num_layers):
        # Layer of parameterized rotations
        for i, qubit in enumerate(qubits):
            # Each qubit gets three rotation parameters (full single-qubit control)
            theta = sympy.Symbol(f'theta_{param_idx}')
            phi = sympy.Symbol(f'phi_{param_idx + 1}')
            lam = sympy.Symbol(f'lambda_{param_idx + 2}')

            params.extend([theta, phi, lam])
            param_idx += 3

            # Apply rotations: RZ(θ)RY(φ)RZ(λ)
            circuit.append([
                cirq.rz(theta)(qubit),
                cirq.ry(phi)(qubit),
                cirq.rz(lam)(qubit)
            ])

        # Layer of entangling gates
        # Use ring topology: connect each qubit to next (and last to first)
        # Ring topology provides all-to-all entanglement in O(n) gates
        # This is a hardware-efficient ansatz suitable for NISQ devices
        for i in range(n_qubits):
            control = qubits[i]
            target = qubits[(i + 1) % n_qubits]
            circuit.append(cirq.CZ(control, target))

    return circuit, params


def compute_expectation_values(
    circuit: cirq.Circuit,
    observables: List[cirq.PauliString]
) -> np.ndarray:
    """
    Compute expectation values of observables on circuit output state.

    In quantum ML, expectation values serve as the classical outputs from
    the quantum layer. These are then fed to classical neural network layers.

    Note: This implementation assumes Hermitian observables. For Hermitian
    operators, expectation values are real. If imaginary part is non-negligible
    (> 1e-6), a warning is issued as this indicates a non-Hermitian observable.

    Args:
        circuit: Quantum circuit (must not be parameterized)
        observables: List of observables to measure (should be Hermitian)

    Returns:
        Array of real expectation values

    Raises:
        ValueError: If circuit is parameterized
        Warning: If observable appears non-Hermitian (has significant imaginary part)
    """
    if cirq.is_parameterized(circuit):
        raise ValueError("Circuit must be resolved before computing expectations")

    simulator = cirq.Simulator()
    expectations = []

    for observable in observables:
        # Compute <ψ|O|ψ> using simulation
        result = simulator.simulate_expectation_values(
            circuit,
            observables=[observable]
        )
        expectation_value = result[0]

        # For Hermitian observables, expectation values must be real
        # Check for significant imaginary component
        imag_part = np.imag(expectation_value)
        if np.abs(imag_part) > 1e-6:
            import warnings
            warnings.warn(
                f"Observable {observable} has non-negligible imaginary expectation "
                f"value {imag_part:.2e}. This suggests the observable is not Hermitian. "
                f"Physical observables must be Hermitian.",
                RuntimeWarning
            )

        expectations.append(np.real(expectation_value))

    return np.array(expectations)


def measure_in_basis(qubit: cirq.Qid, basis: str) -> cirq.Circuit:
    """
    Prepare measurement circuit in specified basis.

    Measuring in different bases requires rotating to that basis first.

    Args:
        qubit: Qubit to measure
        basis: Measurement basis ('Z', 'X', or 'Y')

    Returns:
        Circuit performing basis rotation and measurement
    """
    circuit = cirq.Circuit()

    # Rotate to measurement basis
    if basis == 'X':
        # H: |+>->|0>, |->->|1>
        circuit.append(cirq.H(qubit))
    elif basis == 'Y':
        # S†H: |+i>->|0>, |-i>->|1>
        circuit.append([cirq.S(qubit) ** -1, cirq.H(qubit)])
    elif basis == 'Z':
        # No rotation needed
        pass
    else:
        raise ValueError(f"Unknown basis: {basis}. Valid bases are: 'X', 'Y', 'Z'")

    circuit.append(cirq.measure(qubit, key='result'))
    return circuit


def quantum_layer(
    qubits: List[cirq.Qid],
    input_data: np.ndarray,
    num_layers: int = 1,
    parameter_values: np.ndarray = None
) -> np.ndarray:
    """
    Execute quantum layer: encode data, apply PQC, measure observables.

    This demonstrates the forward pass of a quantum layer in a hybrid model.

    Gradient Computation:
        In full TFQ implementation, gradients w.r.t. quantum parameters are
        computed using the parameter-shift rule. For a rotation gate R(θ):
            ∂⟨O⟩/∂θ = (⟨O⟩_{θ+π/2} - ⟨O⟩_{θ-π/2}) / 2
        This requires two circuit evaluations per parameter but works on
        quantum hardware without needing backpropagation through the circuit.
        Reference: https://pennylane.ai/qml/glossary/parameter_shift.html

    Args:
        qubits: Qubits for the quantum circuit
        input_data: Classical input features
        num_layers: Number of PQC layers
        parameter_values: Values for PQC parameters (if None, use random)

    Returns:
        Classical output vector (expectation values)
    """
    # Step 1: Encode input data
    encoding_circuit = angle_encode_data(qubits, input_data[:len(qubits)])

    # Step 2: Build and resolve PQC
    pqc, params = build_pqc(qubits, num_layers)

    if parameter_values is None:
        # Initialize with random parameters
        parameter_values = np.random.uniform(0, 2 * np.pi, size=len(params))

    param_dict = {param: val for param, val in zip(params, parameter_values)}
    resolved_pqc = cirq.resolve_parameters(pqc, param_dict)

    # Step 3: Combine circuits
    full_circuit = encoding_circuit + resolved_pqc

    # Step 4: Define observables (Z on each qubit)
    observables = [cirq.Z(q) for q in qubits]

    # Step 5: Compute expectation values
    expectations = compute_expectation_values(full_circuit, observables)

    return expectations


def build_hybrid_model(num_qubits: int = 4, num_layers: int = 2) -> Dict[str, Any]:
    """
    Build hybrid quantum-classical model architecture.

    In TFQ, this would return a Keras model. Here we return a description
    showing the architecture conceptually.

    Args:
        num_qubits: Number of qubits in quantum layer
        num_layers: Number of PQC layers

    Returns:
        Dictionary describing the model architecture
    """
    qubits = cirq.LineQubit.range(num_qubits)

    # Build quantum circuit components
    pqc, params = build_pqc(qubits, num_layers)
    observables = [cirq.Z(q) for q in qubits]

    model_description = {
        'data_encoding': 'angle_encoding',
        'pqc_circuit': pqc,
        'num_layers': num_layers,
        'num_qubits': num_qubits,
        'num_parameters': len(params),
        'observables': observables,
        'output_dimension': len(observables),
        'architecture': [
            '1. Input Layer: Classical features',
            '2. Data Encoding: Angle encoding (RY rotations)',
            '3. Quantum Layer: Parameterized quantum circuit',
            f'   - {num_layers} layers of rotations + entanglement',
            f'   - {len(params)} trainable parameters',
            '4. Measurement: Expectation values of Z observables',
            f'5. Output: {len(observables)} classical values in [-1, 1]',
            '6. Classical Layer: Dense layer with sigmoid activation',
            '7. Final Output: Binary classification probability'
        ]
    }

    return model_description


def classify_binary(quantum_output: np.ndarray, weights: np.ndarray = None) -> float:
    """
    Apply classical layer for binary classification.

    Takes quantum expectation values and produces classification probability.

    Args:
        quantum_output: Expectation values from quantum layer
        weights: Weights for classical layer (if None, use equal weights)

    Returns:
        Classification probability in [0, 1]
    """
    if weights is None:
        # Simple average of quantum outputs
        weights = np.ones(len(quantum_output)) / len(quantum_output)

    # Linear combination
    linear_output = np.dot(weights, quantum_output)

    # Numerically stable sigmoid activation: maps R -> [0, 1]
    # Avoids overflow for large negative values by using different formulas
    # for positive and negative inputs
    if linear_output >= 0:
        probability = 1 / (1 + np.exp(-linear_output))
    else:
        exp_x = np.exp(linear_output)
        probability = exp_x / (1 + exp_x)

    return probability


def main():
    """
    Demonstrate hybrid quantum-classical binary classification.

    Shows the complete workflow without full TFQ training:
    1. Generate toy dataset
    2. Build quantum circuits
    3. Extract quantum features
    4. Apply classical layer for classification
    """
    _print_major_header("HYBRID QUANTUM-CLASSICAL BINARY CLASSIFICATION")
    print()
    print(get_implementation_notes())
    print()

    # Setup
    np.random.seed(42)
    num_qubits = 4
    num_samples = 8

    print(f"Configuration:")
    print(f"  - Qubits: {num_qubits}")
    print(f"  - Training samples: {num_samples}")
    print()

    # Generate toy dataset (simplified 2-class problem)
    # Class 0: small feature values
    # Class 1: large feature values
    X_class0 = np.random.uniform(0, np.pi/2, size=(num_samples//2, num_qubits))
    X_class1 = np.random.uniform(np.pi/2, np.pi, size=(num_samples//2, num_qubits))

    X = np.vstack([X_class0, X_class1])
    y = np.array([0] * (num_samples//2) + [1] * (num_samples//2))

    print("Dataset generated:")
    print(f"  Class 0 samples: {len(X_class0)} (features in [0, π/2])")
    print(f"  Class 1 samples: {len(X_class1)} (features in [π/2, π])")
    print()

    # Build model
    model_info = build_hybrid_model(num_qubits=num_qubits, num_layers=2)

    print("Hybrid Model Architecture:")
    for step in model_info['architecture']:
        print(f"  {step}")
    print()
    print(f"Total trainable parameters: {model_info['num_parameters']}")
    print()

    # Demonstrate forward pass
    _print_section_header("FORWARD PASS DEMONSTRATION")
    print()

    qubits = cirq.LineQubit.range(num_qubits)

    # Process a few samples
    for i in range(min(3, num_samples)):
        print(f"Sample {i+1} (True label: {y[i]}):")
        print(f"  Input features: {X[i]}")

        # Quantum layer forward pass
        quantum_features = quantum_layer(qubits, X[i], num_layers=2)
        print(f"  Quantum features: {quantum_features}")

        # Classical layer
        prediction = classify_binary(quantum_features)
        print(f"  Prediction probability: {prediction:.4f}")
        predicted_class = 1 if prediction > 0.5 else 0
        print(f"  Predicted class: {predicted_class}")
        print()

    # Visualize PQC structure
    _print_section_header("PARAMETERIZED QUANTUM CIRCUIT STRUCTURE")
    print()

    pqc, params = build_pqc(qubits, num_layers=2)
    print(f"Circuit depth: {len(pqc)}")
    print(f"Number of parameters: {len(params)}")
    print()
    print("Circuit diagram (first 20 moments):")
    print(pqc[:20])
    print()

    # Show what full training would involve
    _print_section_header("TRAINING PROCESS (Conceptual)")
    print()
    print("With TensorFlow Quantum, training would proceed as:")
    print()
    print("1. Forward Pass:")
    print("   - Encode classical data into quantum states")
    print("   - Apply parameterized quantum circuit")
    print("   - Measure expectation values -> quantum features")
    print("   - Apply classical layers -> predictions")
    print()
    print("2. Compute Loss:")
    print("   - Compare predictions to true labels")
    print("   - Calculate binary cross-entropy loss")
    print()
    print("3. Backward Pass:")
    print("   - Compute gradients of loss w.r.t. quantum parameters")
    print("   - TFQ uses parameter-shift rule for quantum gradients")
    print("   - Backpropagate through classical layers")
    print()
    print("4. Update Parameters:")
    print("   - Adjust quantum circuit parameters")
    print("   - Adjust classical layer weights")
    print("   - Repeat until convergence")
    print()

    # Show encoding strategies
    _print_section_header("DATA ENCODING STRATEGIES")
    print()

    print("1. Angle Encoding (used above):")
    print("   - Maps each feature to a rotation angle")
    print("   - Simple and hardware-efficient")
    print("   - Circuit depth: O(n) for n features")
    print()

    # Demonstrate angle encoding
    test_features = [np.pi/4, np.pi/3, np.pi/2]
    angle_circuit = angle_encode_data(qubits[:3], test_features)
    print(f"   Example - encoding {test_features}:")
    print("   " + str(angle_circuit).replace('\n', '\n   '))
    print()

    print("2. Amplitude Encoding:")
    print("   - Maps features to quantum state amplitudes")
    print("   - Exponentially compact (2^n amplitudes for n qubits)")
    print("   - Requires complex state preparation circuits")
    print()

    # Demonstrate amplitude encoding (simple case)
    data_2d = np.array([0.6, 0.8])  # Already normalized
    amp_circuit = amplitude_encode_data(qubits[:1], data_2d)
    print(f"   Example - encoding normalized {data_2d}:")
    print("   " + str(amp_circuit).replace('\n', '\n   '))
    print()

    # Quantum advantage discussion
    _print_section_header("QUANTUM ADVANTAGE IN MACHINE LEARNING")
    print()
    print("Potential advantages of quantum ML:")
    print()
    print("1. Feature Space:")
    print("   - Quantum states live in exponentially large Hilbert space")
    print("   - May access feature representations unreachable classically")
    print()
    print("2. Kernel Methods:")
    print("   - Quantum kernels can be hard to compute classically")
    print("   - Enables quantum advantage for certain learning tasks")
    print()
    print("3. Hardware Efficiency:")
    print("   - Native quantum operations may be more efficient")
    print("   - Potential for quantum speedup on specific problems")
    print()
    print("Current limitations (NISQ era):")
    print("   - Limited qubit count and circuit depth")
    print("   - Noise degrades quantum advantage")
    print("   - Classical simulation often competitive for small problems")
    print("   - Theoretical advantage not yet demonstrated practically")
    print()

    _print_major_header("For full implementation with trainable models, install TensorFlow Quantum")


if __name__ == "__main__":
    # Check TFQ availability
    if not is_tfq_available():
        print("TensorFlow Quantum is not installed.")
        print("Running simplified demonstration without TFQ integration.")
        print()

    # Run demonstration
    main()
