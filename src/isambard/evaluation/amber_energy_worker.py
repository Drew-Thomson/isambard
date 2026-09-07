import sys
import os
import argparse
import traceback
import tempfile
import io

# Set thread limits IMMEDIATELY to prevent oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from openmm.app import ForceField, Simulation, CutoffNonPeriodic, HBonds, PDBFile, Modeller, Topology, element
from pdbfixer import PDBFixer
import openmm as mm
from openmm import Platform, LangevinMiddleIntegrator, unit
from openmm.unit import kelvin, picosecond, picoseconds, kilojoule, mole, nanometer, kilojoules_per_mole
from openmm.vec3 import Vec3
import numpy as np

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

def evaluate_energy(pdb_content, is_cyclic=False):
    res_map = {
        'DSG': 'ASN', 'DAS': 'ASP', 'DGL': 'GLU', 'DAL': 'ALA', 'DCY': 'CYS',
        'DPN': 'PHE', 'DHI': 'HIS', 'DIL': 'ILE', 'DLY': 'LYS', 'DLE': 'LEU',
        'MED': 'MET', 'DPR': 'PRO', 'DGN': 'GLN', 'DAR': 'ARG', 'DSN': 'SER',
        'DTH': 'THR', 'DVA': 'VAL', 'DTR': 'TRP', 'DTY': 'TYR'
    }
    
    lines = []
    for line in pdb_content.splitlines():
        if line.startswith('TER'):
            continue
        if line.startswith('ATOM') or line.startswith('HETATM'):
            res_name = line[17:20].strip()
            if res_name in res_map:
                new_name = res_map[res_name].ljust(3)
                line = line[:17] + new_name + line[20:]
        
        # Ensure lines have newline characters for stringio
        if not line.endswith('\n'):
            line += '\n'
        lines.append(line)
    
    pdb_str = "".join(lines)
    
    try:
        pdb = PDBFile(io.StringIO(pdb_str))
        topology = pdb.topology
        positions = pdb.positions

        normalize_residues(topology)
        modeller = Modeller(topology, positions)

        # Explicitly enforce peptide bonds between adjacent residues in the same chain
        # to prevent broken chains when coordinates are perturbed beyond PDBFile's distance cutoff
        existing_bonds = set()
        for b in modeller.topology.bonds():
            existing_bonds.add((b[0], b[1]))
            existing_bonds.add((b[1], b[0]))
            
        for chain in modeller.topology.chains():
            chain_residues = list(chain.residues())
            for i in range(len(chain_residues) - 1):
                res1 = chain_residues[i]
                res2 = chain_residues[i+1]
                c_atoms = [a for a in res1.atoms() if a.name == 'C']
                n_atoms = [a for a in res2.atoms() if a.name == 'N']
                if c_atoms and n_atoms:
                    c_atom = c_atoms[0]
                    n_atom = n_atoms[0]
                    if (c_atom, n_atom) not in existing_bonds:
                        modeller.topology.addBond(c_atom, n_atom)
                        existing_bonds.add((c_atom, n_atom))
                        existing_bonds.add((n_atom, c_atom))

        if is_cyclic:
            residues = list(modeller.topology.residues())
            if not residues:
                raise ValueError("No residues found in topology")
            
            if residues:
                n_term_n_list = [atom for atom in residues[0].atoms() if atom.name == 'N']
                c_term_c_list = [atom for atom in residues[-1].atoms() if atom.name == 'C']
                
                if not n_term_n_list:
                    raise ValueError(f"No N atom in first residue {residues[0].name}. Atoms: {[a.name for a in residues[0].atoms()]}")
                if not c_term_c_list:
                    raise ValueError(f"No C atom in last residue {residues[-1].name}. Atoms: {[a.name for a in residues[-1].atoms()]}")
                
                n_term_n = n_term_n_list[0]
                c_term_c = c_term_c_list[0]
                modeller.topology.addBond(n_term_n, c_term_c)
                
            modeller.delete([a for a in modeller.topology.atoms() if a.name == 'OXT'])
        else:
            new_topology = Topology()
            new_positions = []
            atom_map = {}
            oxt_bonds = []
            
            for chain in modeller.topology.chains():
                new_chain = new_topology.addChain(chain.id)
                chain_residues = list(chain.residues())
                for res_idx, res in enumerate(chain_residues):
                    new_res = new_topology.addResidue(res.name, new_chain, res.id, res.insertionCode)
                    for atom in res.atoms():
                        new_atom = new_topology.addAtom(atom.name, atom.element, new_res)
                        atom_map[atom] = new_atom
                        new_positions.append(modeller.positions[atom.index].value_in_unit(unit.nanometer))
                    
                    # Enforce OXT on the C-terminus of EVERY chain for non-cyclic peptides
                    if res_idx == len(chain_residues) - 1:
                        has_oxt = any(a.name == 'OXT' for a in res.atoms())
                        ca_atoms = [a for a in res.atoms() if a.name == 'CA']
                        c_atoms = [a for a in res.atoms() if a.name == 'C']
                        o_atoms = [a for a in res.atoms() if a.name == 'O']
                        
                        if not has_oxt and ca_atoms and c_atoms and o_atoms:
                            ca_atom = ca_atoms[0]
                            c_atom = c_atoms[0]
                            o_atom = o_atoms[0]
                            
                            ca_pos = modeller.positions[ca_atom.index].value_in_unit(unit.nanometer)
                            c_pos = modeller.positions[c_atom.index].value_in_unit(unit.nanometer)
                            o_pos = modeller.positions[o_atom.index].value_in_unit(unit.nanometer)
                            
                            ca_arr = np.array([ca_pos.x, ca_pos.y, ca_pos.z])
                            c_arr = np.array([c_pos.x, c_pos.y, c_pos.z])
                            o_arr = np.array([o_pos.x, o_pos.y, o_pos.z])
                            
                            v = o_arr - ca_arr
                            v1 = c_arr - ca_arr
                            n = np.cross(v1, v)
                            p = np.cross(v, n)
                            
                            p_norm = p / np.linalg.norm(p)
                            oxt_arr = c_arr + p_norm * 0.13
                            
                            oxt_pos = Vec3(oxt_arr[0], oxt_arr[1], oxt_arr[2])
                            new_oxt = new_topology.addAtom('OXT', element.oxygen, new_res)
                            new_positions.append(oxt_pos)
                            oxt_bonds.append((atom_map[c_atom], new_oxt))

            for bond in modeller.topology.bonds():
                new_topology.addBond(atom_map[bond[0]], atom_map[bond[1]])
                
            for b0, b1 in oxt_bonds:
                new_topology.addBond(b0, b1)
                
            new_topology.setPeriodicBoxVectors(modeller.topology.getPeriodicBoxVectors())
            modeller = Modeller(new_topology, new_positions * unit.nanometer)

        forcefield = ForceField('amber14-all.xml', 'implicit/obc1.xml')
        
        if is_cyclic:
            modeller.addHydrogens(forcefield=None)
            modeller.delete([a for a in list(modeller.topology.residues())[0].atoms() if a.name == 'H2' or a.name == 'H3'])
        else:
            modeller.addHydrogens(forcefield=forcefield)
            
        # Rebuild cache if atom counts mismatch
        if cache.simulation is not None:
            if len(modeller.positions) != cache.simulation.context.getSystem().getNumParticles():
                cache.simulation = None
                
        if cache.simulation is None:
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

            try:
                platform = Platform.getPlatformByName('CUDA')
                properties = {'Precision': 'mixed'}
                integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)
                cache.simulation = Simulation(modeller.topology, system, integrator, platform, properties)
            except Exception:
                platform = Platform.getPlatformByName('CPU')
                integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)
                cache.simulation = Simulation(modeller.topology, system, integrator, platform)
            
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
        
    except Exception as e:
        raise e

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "--persistent":
        # Fallback for old direct call
        parser = argparse.ArgumentParser()
        parser.add_argument("pdb_path")
        parser.add_argument("--cyclic", action="store_true")
        args = parser.parse_args()
        try:
            with open(args.pdb_path, 'r') as f:
                pdb_content = f.read()
            e = evaluate_energy(pdb_content, args.cyclic)
            print(e)
        except Exception as e:
            traceback.print_exc()
            sys.stderr.write(f"Worker Error: {str(e)}\n")
            sys.exit(1)
    else:
        # Persistent server mode
        while True:
            line = sys.stdin.readline()
            if not line: break
            line = line.strip()
            if not line: continue
            if line == 'EXIT':
                break
            
            parts = line.split()
            if parts[0] == 'EVALUATE':
                pdb_path = parts[1]
                is_cyclic = parts[2].lower() == 'true'
                try:
                    with open(pdb_path, 'r') as f:
                        pdb_content = f.read()
                    energy = evaluate_energy(pdb_content, is_cyclic)
                    print(f"RESULT {energy}")
                    sys.stdout.flush()
                except Exception as e:
                    traceback.print_exc(file=sys.stderr)
                    msg = str(e).replace('\n', ' ')
                    print(f"ERROR {msg}")
                    sys.stdout.flush()
            elif parts[0] == 'EVALUATE_STRING':
                is_cyclic = parts[1].lower() == 'true'
                pdb_lines = []
                while True:
                    pdb_line = sys.stdin.readline()
                    if not pdb_line: break
                    if pdb_line.strip() == 'END_PDB':
                        break
                    pdb_lines.append(pdb_line)
                pdb_content = "".join(pdb_lines)
                try:
                    energy = evaluate_energy(pdb_content, is_cyclic)
                    print(f"RESULT {energy}")
                    sys.stdout.flush()
                except Exception as e:
                    traceback.print_exc(file=sys.stderr)
                    msg = str(e).replace('\n', ' ')
                    print(f"ERROR {msg}")
                    sys.stdout.flush()
