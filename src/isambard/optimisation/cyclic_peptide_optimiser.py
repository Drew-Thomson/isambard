import copy
import random
import numpy
import ampal
import matplotlib.pyplot as plt
from isambard.specifications.cyclic_peptide import CyclicPeptide, calc_rmsd
from isambard.modelling.daspr import pack_side_chains_daspr
from isambard.evaluation.amber_energy import AmberEnergyEvaluator
from isambard.modelling.kinematic_closure import new_input_angles
from ampal.geometry import dihedral

# Helper for filter_by_rama_rmsd
def toroidal_dist(point1, point2):
    xdiff = abs(point1[0] - point2[0])
    if xdiff > 180: xdiff = 360 - xdiff
    ydiff = abs(point1[1] - point2[1])
    if ydiff > 180: ydiff = 360 - ydiff
    return numpy.sqrt(xdiff**2 + ydiff**2)

def calc_rmsd2(rama1, rama2):
    dist = [toroidal_dist(x, y) for x, y in zip(rama1, rama2)]
    return numpy.sqrt(sum([x**2 for x in dist])/len(dist))

class CyclicPeptideOptimiser:
    """Optimises a cyclic peptide structure."""
    
    def __init__(self, sequence):
        self.sequence = sequence
        self.model = None
        self.evaluator = AmberEnergyEvaluator()
        self.halloffame = []
        self.working_pop = []
        self.energies = []
        self.n_indices = None
        self.ca_indices = None
        self.c_indices = None
        
    def get_energy(self, model):
        """Calculates energy of the model."""
        return self.evaluator(model)
        
    def build_start_mac(self):
        """Builds a starting cyclic peptide structure."""
        # Generating the macrocycle
        # By setting auto_build=True, the CyclicPeptide builds itself on init.
        self.start_mac = CyclicPeptide(self.sequence, auto_build=True)
        
        # Pack side chains
        self.model = pack_side_chains_daspr(self.start_mac, [self.sequence])
        self.model.tags['cyclic'] = True
        
        # Setup residues for filter_by_rama_rmsd
        self.residues = list(self.model[0])
        
    def filter_by_rama_rmsd(self, population, rmsd_val):
        ramalist = []
        for _, model in population:
            # Re-tag torsion angles to ensure they are current
            model[0].tag_torsion_angles()
            
            # Extract phi, psi from tags
            temp_ramas = []
            for res in list(model[0]):
                phi = res.tags.get('phi')
                if phi is None:
                    phi = 0.0
                psi = res.tags.get('psi')
                if psi is None:
                    psi = 0.0
                temp_ramas.append((phi, psi))
            # Shift to match expected format
            rama = temp_ramas[1:] + [temp_ramas[0]]
            ramalist.append(rama)

        rmsd_array = numpy.zeros((len(population), len(population)))
        for j in range(len(population)):
            for k in range(0, j):
                rmsd_array[k][j] = calc_rmsd2(ramalist[j], ramalist[k])
        
        tmp_current = []
        for j in range(len(population)):
            line = rmsd_array[:j, j]
            if all(l > rmsd_val for l in line):
                tmp_current.append(population[j])
        return tmp_current

    def optimise(self, n_iter, samplesize=200, hof_len=5, rama_rmsd=15, plot=True):
        """Optimisation loop."""
        print(f'optimising sequence {self.sequence}')
        
        self.build_start_mac()
        initial_energy = self.get_energy(self.model)
        self.energies = [initial_energy]
        
        # Store as (energy, model)
        self.halloffame = [(initial_energy, copy.deepcopy(self.model))]
        self.working_pop = [(initial_energy, copy.deepcopy(self.model))]
        
        for i in range(n_iter):
            population_results = []
            
            # Simple perturbation loop
            while len(population_results) < samplesize:
                parent_entry = random.choice(self.working_pop)
                parent_model = parent_entry[1]
                
                # Kinematic closure
                indices = sorted(random.sample(range(len(self.sequence)), 3))
                alternate_positions = new_input_angles(parent_model[0], *indices)
                
                for pos in alternate_positions:
                    # Create a new assembly from the new backbone structure
                    new_model = ampal.Assembly([pos])
                    new_model.sequence = self.sequence
                    new_model.tags['cyclic'] = True
                    
                    # Pack side chains
                    try:
                        packed = pack_side_chains_daspr(new_model, [self.sequence])
                        packed.tags['cyclic'] = True
                        energy = self.get_energy(packed)
                        population_results.append((energy, packed))
                    except:
                        continue
            
            # Merge, Sort, Filter
            all_models = self.halloffame + population_results
            all_models.sort(key=lambda x: x[0])
            
            # Apply filtering
            filtered_models = self.filter_by_rama_rmsd(all_models, rama_rmsd)
            
            # Update Hall of Fame and Working Population
            self.halloffame = filtered_models[:hof_len]
            self.working_pop = filtered_models[:samplesize]
            
            # Record best energy
            self.energies.append(self.halloffame[0][0])
            
            if i % 5 == 0:
                print(f"Iteration {i}: Best energy {self.halloffame[0][0]}")

        if plot:
            plt.plot(range(len(self.energies)), self.energies)
            plt.show()
