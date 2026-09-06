import subprocess
import tempfile
import os
import sys
from isambard.modelling.daspr import pack_side_chains_daspr

class AmberEnergyEvaluator:
    """Evaluates the potential energy of a model using OpenMM/CUDA via a persistent worker."""

    def __init__(self, forcefield='amber14-all.xml', solvent='implicit/obc1.xml'):
        self.forcefield = forcefield
        self.solvent = solvent
        self.worker_process = None

    def _start_worker(self):
        worker_script = os.path.join(os.path.dirname(__file__), 'amber_energy_worker.py')
        self.worker_process = subprocess.Popen(
            [sys.executable, worker_script, '--persistent'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

    def __call__(self, model):
        if self.worker_process is None or self.worker_process.poll() is not None:
            self._start_worker()
            
        # 0. Pack side chains if possible (handles D-amino acids)
        try:
            model = pack_side_chains_daspr(model, [model.sequence])
        except Exception:
            pass
        
        # Path to the isolated worker script
        is_cyclic = model.tags.get('cyclic', False)
        
        # 1. Handle model (extract PDB string)
        pdb_string = model.pdb
        fd, pdb_path = tempfile.mkstemp(suffix='.pdb')
        with os.fdopen(fd, 'w') as f:
            f.write(pdb_string)
            
        try:
            # 2. Execute worker via pipe
            self.worker_process.stdin.write(f"EVALUATE {pdb_path} {is_cyclic}\n")
            self.worker_process.stdin.flush()
            
            line = self.worker_process.stdout.readline().strip()
            if line.startswith("RESULT"):
                return float(line.split()[1])
            elif line.startswith("ERROR"):
                raise RuntimeError(f"AmberEnergy evaluation failed in worker: {line[6:]}")
            else:
                err_msg = self.worker_process.stderr.read()
                raise RuntimeError(f"Unexpected worker output: {line}\nStderr: {err_msg}")
            
        finally:
            if os.path.exists(pdb_path):
                os.remove(pdb_path)

    def __del__(self):
        if self.worker_process is not None and self.worker_process.poll() is None:
            try:
                self.worker_process.stdin.write("EXIT\n")
                self.worker_process.stdin.flush()
                self.worker_process.terminate()
                self.worker_process.wait(timeout=1.0)
            except Exception:
                pass

