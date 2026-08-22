import subprocess
import tempfile
import os
import sys

class AmberEnergyEvaluator:
    """Evaluates the potential energy of a model using OpenMM/CUDA."""

    def __init__(self, forcefield='amber14-all.xml', solvent='implicit/obc1.xml'):
        self.forcefield = forcefield
        self.solvent = solvent

    def __call__(self, model):
        # Path to the isolated worker script
        worker_script = os.path.join(os.path.dirname(__file__), 'amber_energy_worker.py')
        
        # 1. Handle model (extract PDB string)
        pdb_string = model.pdb
        fd, pdb_path = tempfile.mkstemp(suffix='.pdb')
        with os.fdopen(fd, 'w') as f:
            f.write(pdb_string)
            
        try:
            # 2. Execute worker script in an isolated subprocess
            # This guarantees 100% resource cleanup and CUDA context isolation.
            result = subprocess.run(
                [sys.executable, worker_script, pdb_path],
                capture_output=True,
                text=True,
                check=True
            )
            # Parse result
            return float(result.stdout.strip())
            
        except subprocess.CalledProcessError as e:
            # Re-raise with descriptive error if the worker script failed
            raise RuntimeError(f"AmberEnergy evaluation failed:\n{e.stderr}")
            
        finally:
            if os.path.exists(pdb_path):
                os.remove(pdb_path)
