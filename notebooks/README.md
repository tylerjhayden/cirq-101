# Cirq-101 Jupyter Notebooks

Interactive Jupyter notebooks for learning quantum computing with Google's Cirq framework.

## Overview

This directory contains comprehensive Jupyter notebooks covering all implemented sections of the Cirq-101 course. Each notebook provides:

- **Interactive code cells** for hands-on learning
- **Detailed explanations** of quantum computing concepts
- **Visualizations** of quantum states, circuits, and results
- **Exercises** for self-guided exploration
- **Complete working examples** you can modify and experiment with

## Notebook Structure

### Part 1: The Cirq Software Development Kit

Fundamentals of quantum circuit construction and simulation.

| Notebook | Topics | Key Concepts |
|----------|--------|--------------|
| [1.2 Building Blocks](part1_section_1_2_building_blocks.ipynb) | Qubits, Gates, Operations | LineQubit, GridQubit, NamedQubit, Gate protocols, Unitary matrices |
| [1.3 Circuits](part1_section_1_3_circuits.ipynb) | Moments, Circuit Construction | Insert strategies, Circuit depth, Parallel operations |
| [1.4 Execution](part1_section_1_4_execution.ipynb) | Simulation and Analysis | run() vs simulate(), Bell states, Expectation values |
| [1.5 Noisy Simulation](part1_section_1_5_noisy_simulation.ipynb) | Realistic Quantum Errors | Noise channels, Density matrices, T1/T2 decay, Purity |

### Part 2: Industry Applications

Real-world quantum algorithms solving practical problems.

| Notebook | Application | Algorithm |
|----------|-------------|-----------|
| [2.1 VQE for H₂](part2_section_2_1_vqe_h2.ipynb) | Quantum Chemistry | Variational Quantum Eigensolver (VQE) |
| [2.2 QAOA Max-Cut](part2_section_2_2_qaoa_maxcut.ipynb) | Combinatorial Optimization | Quantum Approximate Optimization Algorithm (QAOA) |
| [2.3 Hybrid QML](part2_section_2_3_tfq_classification.ipynb) | Machine Learning | Parameterized Quantum Circuits (PQC) |

### Part 3: Synthesis and Best Practices

Advanced topics, design patterns, and professional practices.

| Notebook | Focus | Content |
|----------|-------|---------|
| [3.1 Comparative Analysis](part3_section_3_1_comparative_analysis.ipynb) | Algorithm Patterns | Unified variational framework, Common design patterns |
| [3.2 Best Practices](part3_section_3_2_best_practices.ipynb) | Circuit Design | Hardware awareness, Parameterization, Modularity, Optimization |

## Getting Started

### Prerequisites

```bash
pip install cirq numpy matplotlib scipy networkx openfermion
```

Optional (for full TensorFlow Quantum support):
```bash
pip install tensorflow>=2.11.0 tensorflow-quantum
```

### Running the Notebooks

1. **Start Jupyter**:
   ```bash
   jupyter notebook
   ```

2. **Navigate** to the `notebooks/` directory

3. **Open** any notebook and execute cells sequentially with `Shift+Enter`

### Recommended Learning Path

**Beginners**: Follow the order 1.2 → 1.3 → 1.4 → 1.5 → 2.1 → 2.2 → 2.3 → 3.1 → 3.2

**Experienced**: Jump directly to Part 2 applications, then review Part 3 for best practices

**Researchers**: Focus on Part 2 algorithms and Part 3 comparative analysis

## Notebook Features

### Interactive Exploration

All notebooks are designed for experimentation:

- **Modify parameters**: Change angles, qubit counts, noise levels
- **Visualize results**: See quantum states, circuits, and measurement outcomes
- **Run complete demos**: Each section includes a `main()` function running the full demonstration
- **Access individual functions**: Import and use specific functions for custom exploration

### Example Usage

```python
# In any notebook, after running initial cells:

# Option 1: Run complete demonstration
main()

# Option 2: Use individual functions interactively
from part1_cirq_sdk.section_1_2_building_blocks import demonstrate_gate_protocols
demonstrate_gate_protocols()

# Option 3: Create your own circuits
import cirq
q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))
print(circuit)
```

## Educational Features

### Part 1: Progressive Complexity

- Start with simple concepts (qubits, gates)
- Build to circuit construction and execution
- Culminate in realistic noisy simulation

### Part 2: Complete Workflows

Each application notebook demonstrates:

1. **Problem formulation**: Classical problem → Quantum Hamiltonian
2. **Circuit design**: Ansatz construction with physical intuition
3. **Optimization**: Classical optimization of quantum parameters
4. **Analysis**: Results visualization and interpretation
5. **Performance**: Comparison with exact solutions

### Part 3: Professional Practices

- **Comparative Analysis**: Understand common patterns across VQE, QAOA, QML
- **Best Practices**: Learn hardware-aware design, parameterization strategies, modularity
- **Optimization Techniques**: Circuit depth reduction, gate cancellation

## Visualizations

The notebooks generate rich visualizations:

- **Gate matrices**: Heatmaps of unitary matrices
- **Circuit diagrams**: ASCII and graphical circuit representations
- **State vectors**: Amplitude and phase visualizations
- **Measurement histograms**: Statistical analysis of quantum measurements
- **Density matrices**: Visualization of mixed quantum states
- **Optimization landscapes**: Parameter space exploration for variational algorithms
- **Energy surfaces**: Molecular potential energy curves (VQE)
- **Graph partitions**: Solution visualization for Max-Cut (QAOA)

## Common Issues and Solutions

### Import Errors

If you encounter import errors:

```python
# Add parent directory to path
import sys
sys.path.append('..')
```

### Missing Dependencies

Install missing packages:

```bash
pip install <package-name>
```

### TensorFlow Quantum

Section 2.3 provides a simplified implementation if TFQ is not installed. For full functionality:

```bash
pip install tensorflow>=2.11.0 tensorflow-quantum
```

### Visualization Issues

If plots don't display:

```python
# Add at top of notebook
%matplotlib inline
import matplotlib.pyplot as plt
```

## Additional Resources

### Related Scripts

Each notebook corresponds to a Python script in the project root:

- `part1_cirq_sdk/section_X_Y_*.py`
- `part2_applications/section_X_Y_*.py`
- `part3_synthesis/section_X_Y_*.py`

These scripts can be run standalone:

```bash
python3 part1_cirq_sdk/section_1_2_building_blocks.py
```

### Course Outline

See `/outline.md` for the complete course structure and learning objectives.

### Guidelines

See `/CLAUDE.md` for development guidelines and quantum computing best practices.

## Contributing

When creating or modifying notebooks:

1. **Maintain interactivity**: Enable experimentation with parameters
2. **Add explanations**: Include markdown cells explaining concepts
3. **Provide exercises**: Suggest modifications for self-directed learning
4. **Test thoroughly**: Verify all cells execute without errors
5. **Keep outputs clean**: Clear output before committing (optional)

## License

See project root for license information.

## Support

For questions or issues:

1. Review the notebook's markdown explanations
2. Check the corresponding Python script
3. Consult `/outline.md` for context
4. Reference Cirq documentation: https://quantumai.google/cirq

---

**Happy Quantum Computing!** 🚀

Start with [Section 1.2: Building Blocks](part1_section_1_2_building_blocks.ipynb)
