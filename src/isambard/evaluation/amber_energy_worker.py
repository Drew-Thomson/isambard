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
        'DSG': 'ASN', 'DAS': 'ASP', 'DGL': 'GLU', 'DAL': 'ALA', 'DCY': 'CYS',
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

class SimulationCache:
    def __init__(self):
        self.simulation = None
        self.restraint_force = None
        self.restrained_atoms = []

cache = SimulationCache()

def evaluate_energy(pdb_path, is_cyclic=False):
    res_map = {
        'DSG': 'ASN', 'DAS': 'ASP', 'DGL': 'GLU', 'DAL': 'ALA', 'DCY': 'CYS',
        'DPN': 'PHE', 'DHI': 'HIS', 'DIL': 'ILE', 'DLY': 'LYS', 'DLE': 'LEU',
        'MED': 'MET', 'DPR': 'PRO', 'DGN': 'GLN', 'DAR': 'ARG', 'DSN': 'SER',
        'DTH': 'THR', 'DVA': 'VAL', 'DTR': 'TRP', 'DTY': 'TYR'
    }
    
    lines = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('TER'):
                continue
            if line.startswith('ATOM') or line.startswith('HETATM'):
                res_name = line[17:20].strip()
                if res_name in res_map:
                    new_name = res_map[res_name].ljust(3)
                    line = line[:17] + new_name + line[20:]
            lines.append(line)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
        tmp.writelines(lines)
        tmp.flush()
        tmp_name = tmp.name
        
    try:
        if is_cyclic:
            pdb = PDBFile(tmp_name)
            topology = pdb.topology
            positions = pdb.positions
        else:
            fixer = PDBFixer(tmp_name)
            fixer.findMissingResidues()
            fixer.findMissingAtoms()
            fixer.addMissingAtoms()
            fixer.addMissingHydrogens(7.0)
            topology = fixer.topology
            positions = fixer.positions

        normalize_residues(topology)
        modeller = Modeller(topology, positions)

        if is_cyclic:
            residues = list(modeller.topology.residues())
            n_term_n = [atom for atom in residues[0].atoms() if atom.name == 'N'][0]
            c_term_c = [atom for atom in residues[-1].atoms() if atom.name == 'C'][0]
            modeller.topology.addBond(n_term_n, c_term_c)
            modeller.addHydrogens()    
            modeller.delete([a for a in list(modeller.topology.residues())[-1].atoms() if a.name == 'OXT'])
            modeller.delete([a for a in list(modeller.topology.residues())[0].atoms() if a.name == 'H2' or a.name == 'H3'])
            
        # Rebuild cache if atom counts mismatch
        if cache.simulation is not None:
            if len(modeller.positions) != cache.simulation.context.getSystem().getNumParticles():
                cache.simulation = None
                
        if cache.simulation is None:
            forcefield = ForceField('amber14-all.xml', 'implicit/obc1.xml')
            system = forcefield.createSystem(modeller.topology,
                                             nonbondedMethod=CutoffNonPeriodic,
                                             constraints=HBonds)

            restraint_force = mm.CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
            restraint_force.addGlobalParameter("k", 100.0 * unit.kilocalories_per_mole / unit.nanometer**2)
            restraint_force.addPerParticleParameter("x0")
            restraint_force.addPerParticleParameter("y0")
            restraint_force.addPerParticleParameter("z0")

            cache.restrained_atoms = []
            atoms = list(modeller.topology.atoms())
            for i, atom in enumerate(atoms):
                if atom.name in ['N', 'CA', 'C']:
                    pos = modeller.positions[i].value_in_unit(unit.nanometers)
                    p_idx = restraint_force.addParticle(i, pos)
                    cache.restrained_atoms.append((p_idx, i))
            system.addForce(restraint_force)
            cache.restraint_force = restraint_force

            platform = Platform.getPlatformByName('CUDA')
            properties = {'Precision': 'mixed'}
            integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)
            cache.simulation = Simulation(modeller.topology, system, integrator, platform, properties)
            
        # Update positions and restraints for the current cache
        cache.simulation.context.setPositions(modeller.positions)
        for p_idx, atom_idx in cache.restrained_atoms:
            pos = modeller.positions[atom_idx].value_in_unit(unit.nanometers)
            cache.restraint_force.setParticleParameters(p_idx, atom_idx, pos)
        cache.restraint_force.updateParametersInContext(cache.simulation.context)

        cache.simulation.minimizeEnergy(tolerance=10*kilojoule/(mole*nanometer), maxIterations=500)
        
        state = cache.simulation.context.getState(getEnergy=True)
        energy = state.getPotentialEnergy()
        return energy.value_in_unit(kilojoules_per_mole)
        
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "--persistent":
        # Fallback for old direct call
        parser = argparse.ArgumentParser()
        parser.add_argument("pdb_path")
        parser.add_argument("--cyclic", action="store_true")
        args = parser.parse_args()
        try:
            e = evaluate_energy(args.pdb_path, args.cyclic)
            print(e)
        except Exception as e:
            traceback.print_exc()
            sys.stderr.write(f"Worker Error: {str(e)}\n")
            sys.exit(1)
    else:
        # Persistent server mode
        for line in sys.stdin:
            line = line.strip()
            if not line: continue
            if line == 'EXIT':
                break
            
            parts = line.split()
            if parts[0] == 'EVALUATE':
                pdb_path = parts[1]
                is_cyclic = parts[2].lower() == 'true'
                try:
                    energy = evaluate_energy(pdb_path, is_cyclic)
                    print(f"RESULT {energy}")
                    sys.stdout.flush()
                except Exception as e:
                    traceback.print_exc(file=sys.stderr)
                    msg = str(e).replace('\n', ' ')
                    print(f"ERROR {msg}")
                    sys.stdout.flush()
