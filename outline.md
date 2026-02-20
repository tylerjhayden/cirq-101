# Programming with Google's Cirq: A Graduate Course

## Part I: The Cirq Software Development Kit

This section covers the core data structures and simulation workflows for quantum programming. You'll progress from ideal quantum circuits to realistic noisy simulations that mimic physical hardware.

### Section 1.1: The Cirq Ecosystem

**Introduction**

Cirq is an open-source Python library for writing, manipulating, and optimizing quantum circuits. It targets Noisy Intermediate-Scale Quantum (NISQ) devices and exposes hardware details rather than hiding them. Achieving strong results on NISQ hardware requires tailoring algorithms to specific device characteristics, constraints, and error profiles. Cirq provides a framework for virtual prototyping where you develop the entire experimental workflow, from circuit construction to analyzing noisy, hardware-constrained results.

**Installation**

Install Cirq through pip in a virtual environment:

```bash
pip install --upgrade pip
pip install cirq
```

Essential companion libraries:

* **qsimcirq**: High-performance simulator for larger circuits
  ```bash
  pip install qsimcirq
  ```

* **OpenFermion**: Bridges quantum chemistry and quantum computers
  ```bash
  pip install openfermion
  ```

* **TensorFlow Quantum**: Builds hybrid quantum-classical ML models
  ```bash
  pip install tensorflow-quantum
  ```

**The Broader Ecosystem**

Cirq forms the core of a modular quantum software stack. It interfaces with hardware backends (Google's processors, IonQ systems), integrates with high-performance libraries (NVIDIA cuQuantum), and translates circuits into other quantum languages (QUIL).

### Section 1.2: Building Blocks: Qubits, Gates, Operations

**Qubits (cirq.Qid)**

A qubit is an abstract identifier for a quantum two-level system, not a quantum state. All qubit types inherit from cirq.Qid:

* **cirq.LineQubit**: Qubits in a one-dimensional line
  ```python
  q0, q1, q2 = cirq.LineQubit.range(3)
  ```

* **cirq.GridQubit**: Qubits on a two-dimensional grid (natural for Google's processors)
  ```python
  qubit = cirq.GridQubit(2, 5)
  ```

* **cirq.NamedQubit**: Qubits identified by descriptive strings
  ```python
  msg_qubit = cirq.NamedQubit("Message")
  alice_qubit = cirq.NamedQubit("Alice")
  ```

**Gates vs. Operations**

Cirq distinguishes between Gates and Operations—enabling reusable, modular code:

* A **cirq.Gate** is an immutable quantum transformation not tied to specific qubits (e.g., cirq.X, cirq.H, cirq.CNOT). Think of it as a factory.

* An **cirq.Operation** applies a Gate to specific qubits, representing a concrete quantum effect.

```python
import cirq

q = cirq.GridQubit(0, 0)

hadamard_gate = cirq.H  # Gate
hadamard_operation = hadamard_gate(q)  # Operation
hadamard_operation_alt = cirq.H.on(q)  # Alternative

print(f"Gate: {hadamard_gate}")
print(f"Operation: {hadamard_operation}")
```

A single Gate generates many Operations throughout a circuit. Both are immutable.

**Gate Protocols**

Cirq endows Gates with capabilities through protocols:

* **Unitary Representation**: Retrieve matrix form
  ```python
  x_matrix = cirq.unitary(cirq.X)
  print(x_matrix)
  ```

* **Parameterized Gates**: Use exponentiation to create gate families. cirq.X**$t$ represents rotation around the X-axis by $t \times \pi$ radians.
  ```python
  sqrt_x_gate = cirq.X**0.5
  ```

* **Invertibility**: Gates raised to power -1 are invertible. cirq.inverse() inverts single operations or sequences.
  ```python
  a, b = cirq.LineQubit.range(2)
  original = [cirq.H(a), cirq.CNOT(a, b)]
  inverse = cirq.inverse(original)
  ```

* **Decomposition**: Break complex gates into simpler operations. Call cirq.decompose recursively until only simple gates remain.

### Section 1.3: Assembling Algorithms: Moments and Circuits

**Moments**

A cirq.Moment represents Operations executed during the same time slice. All operations must act on disjoint qubits, modeling quantum hardware parallelism.

**Circuits**

A cirq.Circuit is an ordered sequence of Moments. Build circuits by appending operations using circuit.append(). The InsertStrategy determines placement:

* **NEW_THEN_INLINE (Default)**: Attempts to add operations to the most recent moment. Creates a new moment if qubits are already in use.

* **EARLIEST**: Searches backward to find the first moment where the operation fits. Slides gates left, creating compact circuits.

```python
q0, q1, q2 = cirq.LineQubit.range(3)

# Default strategy
circuit_new = cirq.Circuit(cirq.H(q0))
circuit_new.append(cirq.H(q1))
circuit_new.append(cirq.X(q0))

# EARLIEST strategy
circuit_earliest = cirq.Circuit()
circuit_earliest.append(cirq.H(q0), strategy=cirq.InsertStrategy.EARLIEST)
circuit_earliest.append(cirq.H(q1), strategy=cirq.InsertStrategy.EARLIEST)
circuit_earliest.append(cirq.X(q0), strategy=cirq.InsertStrategy.EARLIEST)
```

EARLIEST excels at circuit compression—critical on NISQ hardware where gates are noisy and decoherence times are short.

**Fundamental Objects**

| Class | Purpose | Example |
|---|---|---|
| cirq.GridQubit | Identifier for a 2D lattice qubit | q = cirq.GridQubit(0, 1) |
| cirq.Gate | Abstract quantum effect | h_gate = cirq.H |
| cirq.Operation | Gate applied to specific qubits | op = cirq.H(q) |
| cirq.Moment | Operations on disjoint qubits in one time slice | cirq.Moment([cirq.H(q0), cirq.X(q1)]) |
| cirq.Circuit | Ordered sequence of moments | cirq.Circuit(moment) |

### Section 1.4: Executing and Analyzing Circuits

**Ideal Simulation**

cirq.Simulator performs noiseless simulation, calculating quantum state vector evolution.

**run() vs. simulate()**

cirq.Simulator offers two execution methods:

* **run()**: Mimics real quantum computers. Executes the circuit and performs measurements, returning classical outcomes only. The final state vector remains inaccessible. The repetitions parameter controls execution count.

* **simulate()**: Returns the full final state vector. This enables inspection of amplitudes and phases—impossible on real hardware but invaluable for verification.

**Bell State Example**

```python
import cirq
import numpy as np
import matplotlib.pyplot as plt

# Define qubits
q0, q1 = cirq.LineQubit.range(2)

# Build circuit - create Bell state
bell_circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CNOT(q0, q1)
)

print("Bell State Circuit:")
print(bell_circuit)

# Initialize simulator
simulator = cirq.Simulator()

# Analyze with simulate()
print("\n--- State Vector Analysis ---")
result_simulate = simulator.simulate(bell_circuit)
print("Final state vector:")
print(np.round(result_simulate.final_state_vector, 3))

# Analyze with run()
bell_circuit.append(cirq.measure(q0, q1, key='result'))

print("\n--- Measurement Sampling ---")
result_run = simulator.run(bell_circuit, repetitions=1000)
counts = result_run.histogram(key='result')
print(f"Measurement results:")
print(counts)

# Visualize
cirq.plot_state_histogram(counts, plt.subplot())
plt.show()
```

The simulate() output confirms the correct entangled state. The run() output demonstrates expected statistics: 50% outcomes are 0 ($|00\rangle$) and 50% are 3 ($|11\rangle$), with outcomes 1 ($|01\rangle$) and 2 ($|10\rangle$) absent.

**Expectation Values**

For variational algorithms (VQE, QAOA), the key quantity is the expectation value of an observable $\langle\psi|O|\psi\rangle$. Cirq's simulate_expectation_values() handles this efficiently.

### Section 1.5: Noisy Simulation

Ideal simulation doesn't reflect NISQ hardware performance. Cirq models noise effects for realistic performance estimates.

**Representing Noise**

Noisy quantum evolution uses quantum channels. The operator-sum representation describes density matrix evolution with Kraus operators $\{A_k\}$:

$$\rho \rightarrow \sum_k A_k \rho A_k^\dagger$$

Kraus operators satisfy $\sum_k A_k^\dagger A_k = I$. One unitary Kraus operator means coherent evolution; multiple operators mean noisy channel.

Common noise channels:

| Channel | Physical Interpretation | Effect |
|---|---|---|
| cirq.bit_flip(p) | Random Pauli X error | $(1-p)\rho + pX\rho X^\dagger$ |
| cirq.depolarize(p) | Symmetric white noise | $(1-p)\rho + \frac{p}{4^n-1}\sum_{P\neq I} P\rho P^\dagger$ |
| cirq.amplitude_damp($\gamma$) | Energy relaxation (T1 decay) | $A_0\rho A_0^\dagger + A_1\rho A_1^\dagger$ |
| cirq.phase_damp($\gamma$) | Phase relaxation (T2 decay) | $A_0\rho A_0^\dagger + A_1\rho A_1^\dagger$ |

**Simulation Strategies**

Two approaches:

* **Density Matrix Simulation**: Tracks the full $2^n \times 2^n$ density matrix. Applies operator-sum formula for each channel. Exact but scales exponentially—feasible only for small qubit counts.

* **Monte Carlo Wavefunction**: More scalable, used by cirq.Simulator and qsimcirq.QSimSimulator. Simulates a single trajectory by probabilistically choosing a Kraus operator at each noisy channel. Repeats many times and averages. Same memory cost as ideal simulation but requires sampling overhead.

**Example**

```python
import cirq
import numpy as np

q = cirq.LineQubit(0)

ideal_circuit = cirq.Circuit(cirq.H(q))
noisy_circuit = cirq.Circuit(cirq.H(q), cirq.amplitude_damp(gamma=0.2).on(q))

simulator = cirq.Simulator()
ideal_result = simulator.simulate(ideal_circuit)
print("Ideal:", np.round(ideal_result.final_state_vector, 3))

density_simulator = cirq.DensityMatrixSimulator()
noisy_result = density_simulator.simulate(noisy_circuit)
print("\nNoisy:", np.round(noisy_result.final_density_matrix, 3))
```

The ideal case yields [0.707, 0.707]. The noisy case shows decayed off-diagonal elements and increased $|0\rangle$ population, reflecting energy relaxation.

**Quantum Virtual Machine (QVM)**

The QVM simulates specific Google hardware (Weber, Rainbow) with realistic noise from device calibration. Using the QVM prepares you for real quantum processors.

Build a QVM by:
* Choosing a processor ID
* Loading device calibration and noise properties
* Packaging specifications and simulator into cirq.Engine

For QVM execution, circuits must be device-ready:
* **Native Gate Set**: Only physically implemented gates
* **Valid Qubits**: Only qubits on device topology
* **Connectivity**: Two-qubit gates only on physically connected qubits

Compilation makes circuits device-ready: transform gates into the native set and map abstract qubits to physical qubits respecting connectivity.

## Part II: Industry Applications

This part applies Cirq to solve complex problems. Each section is a self-contained tutorial demonstrating the complete workflow from problem formulation to results.

### Section 2.1: Quantum Chemistry: H$_2$ Ground State Energy with VQE

The Variational Quantum Eigensolver (VQE) finds molecular ground state energy—fundamental in quantum chemistry.

**Theory**

VQE leverages the variational principle: the expectation value of a Hamiltonian $H$ with any trial wavefunction $|\psi(\vec{\theta})\rangle$ is at least the true ground state energy:

$$\langle\psi(\vec{\theta})|H|\psi(\vec{\theta})\rangle \geq E_g$$

VQE uses two processors:
* **Quantum**: Prepares parameterized trial state $|\psi(\vec{\theta})\rangle$ and measures Hamiltonian expectation
* **Classical**: Runs optimization to adjust parameters $\vec{\theta}$, minimizing energy

**OpenFermion**

OpenFermion translates from electrons and orbitals into qubits and gates, automating molecular Hamiltonian transformations.

**Implementation**

**1. Define Molecule**

```python
import cirq
import openfermion
import openfermionpyscf
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

def create_h2_molecule(bond_length):
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., bond_length))]
    return openfermion.MolecularData(geometry, 'sto-3g', 1, 0)
```

**2. Generate Hamiltonian**

```python
def get_h2_hamiltonian(bond_length):
    h2_molecule = create_h2_molecule(bond_length)
    h2_hamiltonian = openfermion.get_molecular_hamiltonian(h2_molecule.get_integrals())
    return h2_hamiltonian, h2_molecule
```

**3. Map to Qubits**

```python
def get_qubit_hamiltonian(fermionic_hamiltonian):
    return openfermion.jordan_wigner(fermionic_hamiltonian)
```

**4. Build Ansatz**

```python
def build_vqe_ansatz(qubits, theta):
    return cirq.Circuit(
        cirq.Y(qubits[0])**theta,
        cirq.X(qubits[1])**theta,
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.CNOT(qubits[1], qubits[2]),
        cirq.Y(qubits[0])**-theta,
        cirq.X(qubits[1])**-theta,
    )

def prepare_hartree_fock(qubits):
    return cirq.Circuit(cirq.X(qubits[0]), cirq.X(qubits[1]))
```

**5. Define Objective**

```python
def objective_function(params, qubits, hamiltonian, simulator):
    theta = params[0]
    circuit = prepare_hartree_fock(qubits) + build_vqe_ansatz(qubits, theta)

    energy = 0.0
    for pauli_string, coefficient in hamiltonian.terms.items():
        if not pauli_string:
            energy += coefficient
            continue

        obs = openfermion.transforms.get_pauli_sum(openfermion.QubitOperator(pauli_string))
        expectation = simulator.simulate_expectation_values(circuit, observables=obs)
        energy += coefficient * np.real(expectation)

    return energy
```

**6. Optimize**

```python
def run_vqe(bond_length):
    fermionic_ham, molecule = get_h2_hamiltonian(bond_length)
    qubit_ham = get_qubit_hamiltonian(fermionic_ham)

    qubits = cirq.LineQubit.range(openfermion.count_qubits(qubit_ham))
    simulator = cirq.Simulator()

    result = scipy.optimize.minimize(
        objective_function,
        [0.0],
        args=(qubits, qubit_ham, simulator),
        method='CG'
    )

    optimal_energy = result.fun
    exact_energy = openfermion.get_ground_state_energy(molecule.get_molecular_hamiltonian())

    return optimal_energy, exact_energy
```

**7. Plot Results**

```python
bond_lengths = np.linspace(0.3, 2.5, 23)
vqe_energies = []
fci_energies = []

print("Running VQE...")
for length in bond_lengths:
    print(f"  Bond length: {length:.2f} Å")
    vqe_e, fci_e = run_vqe(length)
    vqe_energies.append(vqe_e)
    fci_energies.append(fci_e)

plt.figure(figsize=(10, 6))
plt.plot(bond_lengths, vqe_energies, 'o-', label='VQE')
plt.plot(bond_lengths, fci_energies, 'x-', label='Exact FCI')
plt.xlabel('Bond Length (Å)')
plt.ylabel('Energy (Hartree)')
plt.title('H$_2$ Potential Energy Surface')
plt.legend()
plt.grid()
plt.show()
```

VQE energies track exact FCI energies across all bond lengths, demonstrating the algorithm's accuracy.

### Section 2.2: Optimization: Max-Cut with QAOA

The Quantum Approximate Optimization Algorithm (QAOA) finds approximate solutions to combinatorial optimization problems in finance, logistics, and network design.

**Theory**

QAOA prepares a parameterized state $|\vec{\gamma}, \vec{\beta}\rangle$ and measures cost Hamiltonian $H_C$ expectation. Start in uniform superposition, then repeatedly apply:

* **Cost Unitary** $U(C, \gamma) = e^{-i\gamma H_C}$: Imprints problem structure
* **Mixer Unitary** $U(B, \beta) = e^{-i\beta H_M}$: Explores solution space

A classical optimizer finds optimal angles maximizing $\langle H_C\rangle$.

**Max-Cut Problem**

Partition graph vertices into two sets maximizing edges between sets. Map to cost Hamiltonian where vertices are qubits:

$$H_C = \sum_{ij} \frac{w_{ij}}{2}(I - Z_i Z_j)$$

**Implementation**

**1. Define Graph**

```python
import cirq
import networkx as nx
import sympy
import numpy as np
import matplotlib.pyplot as plt

def create_graph():
    G = nx.Graph()
    G.add_weighted_edges_from([
        (0, 1, 5.0), (0, 3, 2.0), (1, 2, 3.0),
        (1, 3, 1.0), (2, 3, 4.0), (2, 4, 6.0), (3, 4, 2.5)
    ])
    return G

graph = create_graph()
```

**2. Build Circuit**

```python
def build_qaoa_circuit(graph, gamma, beta):
    qubits = sorted([cirq.LineQubit(i) for i in graph.nodes()])
    circuit = cirq.Circuit(cirq.H.on_each(*qubits))

    # Cost layer
    for u, v, data in graph.edges(data=True):
        weight = data['weight']
        circuit.append([
            cirq.CNOT(qubits[u], qubits[v]),
            cirq.rz(2 * gamma * weight).on(qubits[v]),
            cirq.CNOT(qubits[u], qubits[v])
        ])

    # Mixer layer
    circuit.append(cirq.rx(-2 * beta).on_each(*qubits))
    circuit.append(cirq.measure(*qubits, key='result'))

    return circuit
```

**3. Define Objective**

```python
def qaoa_objective(params, graph, simulator):
    gamma_val, beta_val = params
    circuit = build_qaoa_circuit(graph, gamma_val, beta_val)

    results = simulator.run(circuit, repetitions=5000)
    measurements = results.measurements['result']

    cost = 0.0
    for sample in measurements:
        for u, v, data in graph.edges(data=True):
            if sample[u] != sample[v]:
                cost += data['weight']

    return -cost / 5000
```

**4. Optimize**

```python
simulator = cirq.Simulator()
gamma_range = np.linspace(0, np.pi, 20)
beta_range = np.linspace(0, np.pi/2, 20)
cost_grid = np.zeros((len(gamma_range), len(beta_range)))

for i, g in enumerate(gamma_range):
    for j, b in enumerate(beta_range):
        cost_grid[i, j] = -qaoa_objective([g, b], graph, simulator)

best_idx = np.unravel_index(np.argmax(cost_grid), cost_grid.shape)
best_gamma = gamma_range[best_idx[0]]
best_beta = beta_range[best_idx[1]]

print(f"Optimal: γ={best_gamma:.3f}, β={best_beta:.3f}")
print(f"Max-Cut: {np.max(cost_grid):.3f}")
```

**5. Visualize Solution**

```python
final_circuit = build_qaoa_circuit(graph, best_gamma, best_beta)
results = simulator.run(final_circuit, repetitions=1000)
counts = results.histogram(key='result')
solution = format(counts.most_common(1)[0][0], f'0{len(graph.nodes())}b')

colors = ['red' if bit == '0' else 'blue' for bit in solution]
nx.draw(graph, with_labels=True, node_color=colors)
plt.title("Max-Cut Partition")
plt.show()
```

### Section 2.3: Machine Learning: Hybrid Classification with TFQ

TensorFlow Quantum builds hybrid quantum-classical models integrating Cirq with TensorFlow.

**PQC Layer**

The tfq.layers.PQC Keras layer:
* Accepts input quantum circuits
* Appends parameterized quantum circuit
* Executes on simulator or hardware
* Calculates observable expectations
* Returns classical values to Keras layers

The layer is differentiable—gradients backpropagate to update quantum parameters.

**Implementation**

**1. Prepare Data**

```python
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

def filter_36(x, y):
    keep = (y == 3) | (y == 6)
    return x[keep], (y[keep] == 3).astype(int)

x_train, y_train = filter_36(x_train, y_train)
x_test, y_test = filter_36(x_test, y_test)

x_train = tf.image.resize(x_train[..., tf.newaxis], (4,4)).numpy() / 255
x_test = tf.image.resize(x_test[..., tf.newaxis], (4,4)).numpy() / 255

def encode_circuit(image):
    pixels = np.ndarray.flatten(image)
    qubits = cirq.GridQubit.rect(4, 4)
    circuit = cirq.Circuit()
    for i, pixel in enumerate(pixels):
        if pixel > 0:
            circuit.append(cirq.rx(np.pi * pixel)(qubits[i]))
    return circuit

x_train_circ = tfq.convert_to_tensor([encode_circuit(x) for x in x_train])
x_test_circ = tfq.convert_to_tensor([encode_circuit(x) for x in x_test])
```

**2. Design PQC**

```python
def create_model_circuit():
    data_qubits = cirq.GridQubit.rect(4, 4)
    readout = cirq.GridQubit(-1, -1)
    params = sympy.symbols('q_0:32')

    circuit = cirq.Circuit()
    for i, qubit in enumerate(data_qubits):
        circuit.append(cirq.ry(params[i])(qubit))

    for i in range(4):
        for j in range(4):
            if i < 3: circuit.append(cirq.CZ(data_qubits[i,j], data_qubits[i+1,j]))
            if j < 3: circuit.append(cirq.CZ(data_qubits[i,j], data_qubits[i,j+1]))

    circuit.append(cirq.ry(params[16])(readout))
    return circuit, cirq.Z(readout)

model_circuit, readout = create_model_circuit()
```

**3. Build Model**

```python
def build_model():
    quantum_input = tf.keras.Input(shape=(), dtype=tf.string)
    expectation = tfq.layers.PQC(model_circuit, readout)(quantum_input)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(expectation)
    return tf.keras.Model(inputs=quantum_input, outputs=output)

model = build_model()
```

**4. Train**

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.01),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train_circ[:500],
    y_train[:500],
    batch_size=32,
    epochs=10,
    validation_data=(x_test_circ[:100], y_test[:100])
)

plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

## Part III: Synthesis and Future Directions

### Section 3.1: Comparative Analysis

Examining VQE, QAOA, and QML reveals a unifying pattern: all three instantiate a variational hybrid quantum-classical loop.

**Unifying Principle**

Use a classical computer to optimize quantum computation parameters:

* **VQE**: Find $\vec{\theta}$ minimizing $\langle H(\vec{\theta})\rangle$ for ground state energy
* **QAOA**: Find $(\vec{\gamma}, \vec{\beta})$ maximizing $\langle H_C(\vec{\gamma}, \vec{\beta})\rangle$ for optimal solutions
* **Quantum ML**: Find $\vec{\theta}$ minimizing $L(\langle O_i(\vec{\theta})\rangle, y)$

All follow: result = classical_optimizer(f(quantum_circuit(params)))

This shifts NISQ algorithm development to engineering domain-specific plugins. Key challenges:

* **Problem Mapping**: Map problems onto quantum objective functions
* **Ansatz Design**: Design circuits expressive yet shallow for noisy hardware

**Comparison**

|  | VQE | QAOA | Quantum ML |
|---|---|---|---|
| **Goal** | Minimum eigenvalue | Approximate optimization solution | Classification/generalization |
| **Quantum Objective** | $\langle H\rangle$ | $\langle H_C\rangle$ | $\langle O_i\rangle$ |
| **Ansatz** | Physically motivated or hardware-efficient | Cost-mixer alternation | Layered PQC |
| **Optimizer** | Minimize $\langle H\rangle$ | Maximize $\langle H_C\rangle$ | Minimize $L(\langle O_i\rangle, y)$ |

### Section 3.2: Advanced Topics and Best Practices

**Beyond Basics**

Advanced tools:

* **Stim**: High-performance QEC code simulator—orders of magnitude faster for Clifford circuits
* **Qualtran**: Analyzes resource requirements for fault-tolerant algorithms

**Best Practices**

* **Hardware Awareness**: Model device constraints early using QVM
* **Parameterization**: Use sympy.Symbol for flexible, reusable circuits
* **Modularity**: Decompose circuits into reusable components
* **Verify First**: Use ideal simulation before expensive noisy simulations
