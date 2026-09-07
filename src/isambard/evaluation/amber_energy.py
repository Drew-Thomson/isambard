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

    def __call__(self, model, pack_side_chains=True):
        if self.worker_process is None or self.worker_process.poll() is not None:
            self._start_worker()
            
        # Extract tag before packing since packing returns a new ampal object without the old tags
        is_cyclic = model.tags.get('cyclic', False)
            
        # 0. Pack side chains if possible (handles D-amino acids)
        if pack_side_chains:
            try:
                model = pack_side_chains_daspr(model, [model.sequence])
                model.tags['cyclic'] = is_cyclic
            except Exception:
                pass
        
        # 1. Handle model (extract PDB string)
        pdb_string = model.pdb
        
        try:
            # 2. Execute worker via pipe (using string interface to avoid disk I/O)
            self.worker_process.stdin.write(f"EVALUATE_STRING {is_cyclic}\n")
            self.worker_process.stdin.write(pdb_string)
            if not pdb_string.endswith('\n'):
                self.worker_process.stdin.write('\n')
            self.worker_process.stdin.write("END_PDB\n")
            self.worker_process.stdin.flush()
            
            line = self.worker_process.stdout.readline().strip()
            if line.startswith("RESULT"):
                return float(line.split()[1])
            elif line.startswith("ERROR"):
                # Non-blocking read of stderr to get traceback
                import time
                time.sleep(0.1)
                import fcntl
                import os
                fl = fcntl.fcntl(self.worker_process.stderr, fcntl.F_GETFL)
                fcntl.fcntl(self.worker_process.stderr, fcntl.F_SETFL, fl | os.O_NONBLOCK)
                err_msg = ""
                try:
                    err_msg = self.worker_process.stderr.read()
                except:
                    pass
                raise RuntimeError(f"AmberEnergy evaluation failed in worker: {line[6:]}\nStderr:\n{err_msg}")
            else:
                err_msg = self.worker_process.stderr.read()
                raise RuntimeError(f"Unexpected worker output: {line}\nStderr: {err_msg}")
            
        except Exception as e:
            raise e

    def __del__(self):
        if self.worker_process is not None and self.worker_process.poll() is None:
            try:
                self.worker_process.stdin.write("EXIT\n")
                self.worker_process.stdin.flush()
                self.worker_process.terminate()
                self.worker_process.wait(timeout=1.0)
            except Exception:
                pass

