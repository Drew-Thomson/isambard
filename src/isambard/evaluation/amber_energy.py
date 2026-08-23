import subprocess
import tempfile
import os
import sys
from isambard.modelling.daspr import pack_side_chains_daspr

class AmberEnergyEvaluator:
    """Evaluates the potential energy of a model using OpenMM/CUDA."""

    def __init__(self, forcefield='amber14-all.xml', solvent='implicit/obc1.xml'):
        self.forcefield = forcefield
        self.solvent = solvent

    def __call__(self, model):
        # 0. Pack side chains if possible (handles D-amino acids)
        try:
            model = pack_side_chains_daspr(model, [model.sequence])
        except Exception:
            # Fallback if packing fails or isn't applicable
            pass
        
        # Path to the isolated worker script
        worker_script = os.path.join(os.path.dirname(__file__), 'amber_energy_worker.py')
        
        # Check if the model is cyclic
        is_cyclic = model.tags.get('cyclic', False)
        
        # 1. Handle model (extract PDB string)
        pdb_string = model.pdb
        fd, pdb_path = tempfile.mkstemp(suffix='.pdb')
        with os.fdopen(fd, 'w') as f:
            f.write(pdb_string)
            
        try:
            # 2. Execute worker script in an isolated subprocess
            # This guarantees 100% resource cleanup and CUDA context isolation.
            cmd = [sys.executable, worker_script, pdb_path]
            if is_cyclic:
                cmd.append('--cyclic')
                
            result = subprocess.run(
                cmd,
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
