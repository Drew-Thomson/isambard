import random
import copy
import ampal
import sys
from ampal.geometry import distance
from .ta_polypeptide import TAPolypeptide

def calc_rmsd(frag1, frag2):
    """Returns the maximum distance between a pair of atoms in equivalent sets."""
    frag1_atoms = frag1.backbone.get_atoms()
    frag2_atoms = frag2.backbone.get_atoms()
    return (0.25 * sum([distance(x, y) for x, y in zip(frag1_atoms, frag2_atoms)]))**0.5

def rand_mac(res, max_iter=20000, max_attempts=5, max_rmsd=0.05):
    """Builds a macrocycle by starting with random phi and psi angles, then mutating them till the ends overlap."""
    initial_phi_psi = [[180, random.uniform(-180, 180), random.uniform(-180, 180)] for _ in range(res)]
    best_angles = initial_phi_psi[:]
    test_model = TAPolypeptide([initial_phi_psi[-1]] + initial_phi_psi + initial_phi_psi[0:2])
    best_rmsd = calc_rmsd(test_model[1], test_model[-2])
    print("Starting rmsd is {0}".format(best_rmsd))
    cached_tries = []
    attempts = 0
    i = 0
    while best_rmsd > max_rmsd and i < max_iter:
        working_angles = copy.deepcopy(best_angles)
        for j in range(random.randint(2,6)):
            working_angles[random.choice(range(len(initial_phi_psi)))][random.choice(range(1,3))] += random.gauss(0,0.5*best_rmsd)

        test_model = TAPolypeptide([working_angles[-1]] + working_angles+working_angles[0:2])
        new_rmsd = calc_rmsd(test_model[1], test_model[-2])
        if (new_rmsd < best_rmsd) or (random.random() < 0.005): 
            best_angles = copy.deepcopy(working_angles)
            best_rmsd = new_rmsd
        if not(i % 100):
            sys.stdout.write("\rAt iter {0} best rmsd is {1})".format(i, best_rmsd))
            sys.stdout.flush()
        i += 1
        if i == max_iter:
            print('\nManaged only rmsd of {0}: resampling!'.format(best_rmsd))
            cached_tries.append((best_angles, best_rmsd))
            if attempts == max_attempts:
                cached_tries.sort(key=lambda x: x[1])
                print("Ran out of attempts, best rmsd was {0}".format(cached_tries[0][1]))
                return(cached_tries[0][0])
            for k in range(len(best_angles)):
                best_angles[k] = [180, random.uniform(-180, 180), random.uniform(-180, 180)]
            test_model = TAPolypeptide([best_angles[-1]]+best_angles+best_angles[0:2])
            best_rmsd = calc_rmsd(test_model[1], test_model[-2])
            attempts += 1
            i = 0

    print("After {0} iterations and {1} attempts best rmsd is {2}".format(i, attempts, best_rmsd))
    return(best_angles)
    
def build_mac(angles):
    """Builds a cyclic peptide in isambard from a list of phi, psi angles."""
    mac = TAPolypeptide([angles[-1]] + angles + [angles[0]])
    mac.tag_torsion_angles()
    actualmac = mac[1:-1]
    actualmac.tags['cyclic'] = True
    return ampal.assembly.Assembly(actualmac)

class CyclicPeptide(ampal.Assembly):
    """Models a cyclic peptide."""

    def __init__(self, sequence, angles=None, auto_build=True):
        super(CyclicPeptide, self).__init__()
        self.sequence = sequence
        self.angles = angles
        if auto_build:
            self.build()

    def build(self):
        """Builds the cyclic peptide using random macrocycle generator or provided angles."""
        if self.angles is None:
            self.angles = rand_mac(len(self.sequence))
        self.actualmac = build_mac(self.angles)
        self._molecules = list(self.actualmac)
        self.tags['cyclic'] = True  # Tag the assembly
        self.relabel_all()
        for m in self._molecules:
            m.ampal_parent = self
