"""Script to create all Jupyter notebooks from Python scripts."""

import json
import sys
from pathlib import Path

# Map sections to their source files
NOTEBOOKS = {
    'part1_section_1_4_execution': {
        'title': 'Section 1.4: Executing and Analyzing Circuits',
        'source': 'part1_cirq_sdk/section_1_4_execution.py',
        'description': 'Learn to execute circuits with cirq.Simulator, understand run() vs simulate(), and analyze quantum states.',
        'key_concepts': ['cirq.Simulator', 'run() vs simulate()', 'Bell states', 'Expectation values', 'Measurement statistics']
    },
    'part1_section_1_5_noisy_simulation': {
        'title': 'Section 1.5: Noisy Simulation',
        'source': 'part1_cirq_sdk/section_1_5_noisy_simulation.py',
        'description': 'Model realistic quantum errors with noise channels and density matrix simulation.',
        'key_concepts': ['Noise channels', 'Density matrices', 'T1/T2 decay', 'Purity', 'Fidelity']
    },
    'part2_section_2_1_vqe_h2': {
        'title': 'Section 2.1: VQE for H₂ Molecule',
        'source': 'part2_applications/section_2_1_vqe_h2.py',
        'description': 'Implement Variational Quantum Eigensolver to compute molecular ground state energy.',
        'key_concepts': ['VQE algorithm', 'OpenFermion', 'Jordan-Wigner', 'Variational principle', 'Quantum chemistry']
    },
    'part2_section_2_2_qaoa_maxcut': {
        'title': 'Section 2.2: QAOA for Max-Cut',
        'source': 'part2_applications/section_2_2_qaoa_maxcut.py',
        'description': 'Apply Quantum Approximate Optimization Algorithm to graph partitioning problems.',
        'key_concepts': ['QAOA', 'Combinatorial optimization', 'Cost Hamiltonian', 'Mixer Hamiltonian', 'Graph problems']
    },
    'part2_section_2_3_tfq_classification': {
        'title': 'Section 2.3: Hybrid Quantum-Classical Machine Learning',
        'source': 'part2_applications/section_2_3_tfq_classification.py',
        'description': 'Build hybrid quantum-classical models for classification tasks.',
        'key_concepts': ['Parameterized Quantum Circuits', 'Data encoding', 'Quantum features', 'Hybrid models', 'TensorFlow Quantum']
    },
    'part3_section_3_1_comparative_analysis': {
        'title': 'Section 3.1: Comparative Analysis of Variational Algorithms',
        'source': 'part3_synthesis/section_3_1_comparative_analysis.py',
        'description': 'Compare VQE, QAOA, and QML to identify common patterns and design principles.',
        'key_concepts': ['Variational framework', 'Problem mapping', 'Ansatz design', 'Optimizer requirements', 'Common patterns']
    },
    'part3_section_3_2_best_practices': {
        'title': 'Section 3.2: Best Practices for Quantum Circuit Design',
        'source': 'part3_synthesis/section_3_2_best_practices.py',
        'description': 'Master hardware-aware design, parameterization, modularity, and optimization techniques.',
        'key_concepts': ['Hardware awareness', 'Parameterization', 'Modularity', 'Circuit optimization', 'NISQ constraints']
    }
}

def create_notebook_structure(notebook_info, source_path, base_dir):
    """Create a notebook structure with appropriate cells."""

    # Read the source Python file
    with open(source_path, 'r') as f:
        source_code = f.read()

    # Extract functions from source
    import re

    # Find all function definitions
    functions = re.findall(r'^def\s+(\w+)\([^)]*\):[^\n]*\n(?:    """[^"]*"""\n)?', source_code, re.MULTILINE)

    cells = []

    # Title cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# {notebook_info['title']}\\n\\n",
            f"{notebook_info['description']}\\n\\n",
            "## Key Concepts\\n\\n",
            "\\n".join([f"- {concept}" for concept in notebook_info['key_concepts']])
        ]
    })

    # Imports cell — module_path is built from the path relative to base_dir
    # so that it matches the package structure when the project root is on sys.path.
    # base_dir is passed in by the caller and has already been validated.
    relative = source_path.relative_to(base_dir)
    module_path = '.'.join(relative.with_suffix('').parts)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": ["# Run the complete demonstration\\n",
                   "import sys\\n",
                   "from pathlib import Path\\n",
                   "sys.path.insert(0, str(Path().resolve().parent))  # Add project root\\n",
                   f"from {module_path} import main\\n\\n",
                   "main()"]
    })

    # Interactive exploration cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Interactive Exploration\\n\\n",
                   "The code above ran the complete demonstration. Now you can import and use individual functions interactively."]
    })

    # Create the notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.9.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    return notebook

def main():
    base_dir = Path(__file__).parent
    notebooks_dir = base_dir / 'notebooks'

    print("Creating Jupyter notebooks...")
    print(f"Output directory: {notebooks_dir}")
    print()

    for notebook_name, notebook_info in NOTEBOOKS.items():
        source_path = base_dir / notebook_info['source']
        output_path = notebooks_dir / f"{notebook_name}.ipynb"

        if not source_path.resolve().is_relative_to(base_dir.resolve()):
            raise ValueError(f"Source path {source_path} escapes base directory {base_dir}")

        if not source_path.exists():
            print(f"⚠ Skipping {notebook_name}: source file not found at {source_path}")
            continue

        print(f"Creating {output_path.name}...")
        notebook = create_notebook_structure(notebook_info, source_path, base_dir)

        with open(output_path, 'w') as f:
            json.dump(notebook, f, indent=2)

        print(f"  ✓ Created successfully")

    print()
    print(f"All notebooks created in {notebooks_dir}")

if __name__ == '__main__':
    main()
