import sys
import os

# Set thread limits IMMEDIATELY to prevent oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from openmm.app import ForceField, Modeller, Simulation, CutoffNonPeriodic, HBonds
from openmm import Platform, LangevinMiddleIntegrator
from openmm.unit import kelvin, picosecond, picoseconds, kilojoule, mole, nanometer, kilojoules_per_mole
from pdbfixer import PDBFixer

def run_worker(pdb_path):
    try:
        # Repair structure
        fixer = PDBFixer(filename=pdb_path)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        
        # Setup ForceField and add Hydrogens
        forcefield = ForceField('amber14-all.xml', 'implicit/obc1.xml')
        modeller = Modeller(fixer.topology, fixer.positions)
        modeller.addHydrogens(forcefield=forcefield)
        
        # Create system
        system = forcefield.createSystem(modeller.topology, 
                                         nonbondedMethod=CutoffNonPeriodic, 
                                         constraints=HBonds)
        
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
        sys.stderr.write(f"Worker Error: {str(e)}\n")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)
    run_worker(sys.argv[1])
