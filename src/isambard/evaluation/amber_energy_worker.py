import sys
import os
import argparse
import traceback

# Set thread limits IMMEDIATELY to prevent oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from openmm.app import ForceField, Modeller, Simulation, CutoffNonPeriodic, HBonds, PDBFile
from openmm import Platform, LangevinMiddleIntegrator, Vec3, System
from openmm.unit import kelvin, picosecond, picoseconds, kilojoule, mole, nanometer, kilojoules_per_mole

def normalize_residues(topology):
    res_map = {
        'DSG': 'SER', 'DAS': 'ASP', 'DGL': 'GLU', 'DAL': 'ALA', 'DCY': 'CYS',
        'DPN': 'PHE', 'DHI': 'HIS', 'DIL': 'ILE', 'DLY': 'LYS', 'DLE': 'LEU',
        'MED': 'MET', 'DPR': 'PRO', 'DGN': 'GLN', 'DAR': 'ARG', 'DSN': 'SER',
        'DTH': 'THR', 'DVA': 'VAL', 'DTR': 'TRP', 'DTY': 'TYR',
        'ASN': 'ASN', 'GLY': 'GLY', 'ALA': 'ALA', 'ASP': 'ASP', 'GLU': 'GLU', 'CYS': 'CYS',
        'PHE': 'PHE', 'HIS': 'HIS', 'ILE': 'ILE', 'LYS': 'LYS', 'LEU': 'LEU',
        'MET': 'MET', 'PRO': 'PRO', 'GLN': 'GLN', 'ARG': 'ARG', 'SER': 'SER',
        'THR': 'THR', 'VAL': 'VAL', 'TRP': 'TRP', 'TYR': 'TYR'
    }
    for res in topology.residues():
        name = res.name.strip()
        if len(name) > 3:
            if name.startswith('N') or name.startswith('C'):
                name = name[1:] if name.startswith('N') else name[:-1]
            else:
                name = name[:3]
        
        normalized_name = res_map.get(name, name[:3])
        res.name = normalized_name

def run_worker(pdb_path, is_cyclic=False):
    print(f"DEBUG: Starting run_worker, is_cyclic={is_cyclic}", file=sys.stderr)
    try:
        # Pre-process: remove TER records
        with open(pdb_path, 'r') as f:
            lines = [line for line in f if not line.startswith('TER')]
        with open(pdb_path, 'w') as f:
            f.writelines(lines)
            
        print("DEBUG: Loading PDB file...", file=sys.stderr)
        pdb = PDBFile(pdb_path)
        
        # Create Modeller
        modeller = Modeller(pdb.topology, pdb.positions)
        normalize_residues(modeller.topology)
        
        # Load forcefield
        forcefield = ForceField('amber14-all.xml', 'implicit/obc1.xml')

        # Add cyclic bond FIRST, before adding hydrogens
        if is_cyclic:
            print("DEBUG: Creating cyclic bond...", file=sys.stderr)
            residues = list(modeller.topology.residues())
            n_term = [at for at in residues[0].atoms() if at.name == 'N'][0]
            c_term = [at for at in residues[-1].atoms() if at.name == 'C'][0]
            modeller.topology.addBond(n_term, c_term)
            
            # Remove N-terminal hydrogens and C-terminal OXT
            print("DEBUG: Removing terminal caps...", file=sys.stderr)
            excess_atoms = []
            for at in residues[0].atoms():
                if at.name in ['H1', 'H2', 'H3']: excess_atoms.append(at)
            for at in residues[-1].atoms():
                if at.name == 'OXT': excess_atoms.append(at)
            modeller.delete(excess_atoms)
        
        # Add Hydrogens
        print("DEBUG: Adding hydrogens...", file=sys.stderr)
        modeller.addHydrogens(pH=7.0)
        print("DEBUG: Hydrogens added.", file=sys.stderr)
        
        # Create system
        # Try to suppress terminal patching by giving ForceField a topology 
        # where the cyclic residues are NOT at the termini
        print("DEBUG: Creating system...", file=sys.stderr)
        
        # Let's see what ForceField thinks are terminal residues
        # Residues are 'GLY'
        
        system = forcefield.createSystem(modeller.topology, 
                                         nonbondedMethod=CutoffNonPeriodic, 
                                         constraints=HBonds)
        
        # Initialize CUDA platform
        platform = Platform.getPlatformByName('CUDA')
        properties = {'Precision': 'mixed'}
        
        # Minimization
        print("DEBUG: Starting minimization...", file=sys.stderr)
        integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)
        simulation = Simulation(modeller.topology, system, integrator, platform, properties)
        simulation.context.setPositions(modeller.positions)
        simulation.minimizeEnergy(tolerance=10*kilojoule/(mole*nanometer), maxIterations=500)
        
        # Energy Calculation
        state = simulation.context.getState(getEnergy=True)
        energy = state.getPotentialEnergy()
        print(energy.value_in_unit(kilojoules_per_mole))
        
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.write(f"Worker Error: {str(e)}\n")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdb_path")
    parser.add_argument("--cyclic", action="store_true")
    args = parser.parse_args()
    run_worker(args.pdb_path, is_cyclic=args.cyclic)
