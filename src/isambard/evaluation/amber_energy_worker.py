import sys
import os
import argparse
import traceback
import tempfile

# Set thread limits IMMEDIATELY to prevent oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from openmm.app import ForceField, Simulation, CutoffNonPeriodic, HBonds, PDBFile, Modeller
from pdbfixer import PDBFixer
import openmm as mm
from openmm import Platform, LangevinMiddleIntegrator, unit
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
    print("--- RUNNING WORKER ---")
    try:
        # Pre-process: remove TER records
        with open(pdb_path, 'r') as f:
            lines = [line for line in f if not line.startswith('TER')]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb') as tmp:
            tmp.writelines(lines)
            tmp.flush()
            if is_cyclic:
                pdb = PDBFile(tmp.name)
                topology = pdb.topology
                positions = pdb.positions
            else:
                fixer = PDBFixer(tmp.name)
                fixer.findMissingResidues()
                fixer.findMissingAtoms()
                fixer.addMissingAtoms()
                fixer.addMissingHydrogens(7.0)
                topology = fixer.topology
                positions = fixer.positions

        normalize_residues(topology)
        modeller = Modeller(topology, positions)

        # Create cyclic bond (N of first residue to C of last)
        if is_cyclic:
            print("Cyclic processing: Adding bond and deleting terminal atoms")
            residues = list(modeller.topology.residues())
            n_term_n = [atom for atom in residues[0].atoms() if atom.name == 'N'][0]
            c_term_c = [atom for atom in residues[-1].atoms() if atom.name == 'C'][0]
            modeller.topology.addBond(n_term_n, c_term_c)
            modeller.addHydrogens()    
            modeller.delete([a for a in [r for r in modeller.topology.residues()][-1].atoms() if a.name == 'OXT'])
            modeller.delete([a for a in [r for r in modeller.topology.residues()][0].atoms() if a.name == 'H2' or a.name == 'H3'])
            
            # Debug: Check residues
            for res in modeller.topology.residues():
                print(f"Residue: {res.name}, Atoms: {[a.name for a in res.atoms()]}")
                    
        # Setup ForceField
        print("Creating ForceField and System...")
        forcefield = ForceField('amber14-all.xml', 'implicit/obc1.xml')
        
        # Create system
        system = forcefield.createSystem(modeller.topology,
                                         nonbondedMethod=CutoffNonPeriodic,
                                         constraints=HBonds)


        # Apply positional restraints to N, CA, C atoms
        restraint_force = mm.CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        restraint_force.addGlobalParameter("k", 100.0 * unit.kilocalories_per_mole / unit.nanometer**2)
        restraint_force.addPerParticleParameter("x0")
        restraint_force.addPerParticleParameter("y0")
        restraint_force.addPerParticleParameter("z0")

        atoms = list(modeller.topology.atoms())
        for i, (atom, pos) in enumerate(zip(atoms, modeller.positions)):
            if atom.name in ['N', 'CA', 'C']:
                restraint_force.addParticle(i, pos.value_in_unit(unit.nanometers))
        system.addForce(restraint_force)

        # Initialize CUDA platform
        platform = Platform.getPlatformByName('CUDA')
        properties = {'Precision': 'mixed'}

        # Minimization
        integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)
        simulation = Simulation(modeller.topology, system, integrator, platform, properties)
        simulation.context.setPositions(modeller.positions)
        simulation.minimizeEnergy(tolerance=10*kilojoule/(mole*nanometer), maxIterations=500)

        # Energy Calculation
        state = simulation.context.getState(getEnergy=True)
        energy = state.getPotentialEnergy()
        print(energy.value_in_unit(kilojoules_per_mole))

    except Exception as e:
        traceback.print_exc()
        sys.stderr.write(f"Worker Error: {str(e)}\n")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdb_path")
    parser.add_argument("--cyclic", action="store_true")
    args = parser.parse_args()
    run_worker(args.pdb_path, is_cyclic=args.cyclic)
