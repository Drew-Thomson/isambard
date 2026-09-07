#!/usr/bin/env python
# coding: utf-8

# # Testing the Fast OpenMM Cyclic Peptide Optimiser
# This notebook tests the updated `CyclicPeptideOptimiser` which runs purely in OpenMM to avoid expensive side-chain packing and PDB roundtripping during the conformational search.

# In[ ]:


import sys
import time
from isambard.optimisation.cyclic_peptide_optimiser import CyclicPeptideOptimiser
from openmm.app import PDBFile
import matplotlib.pyplot as plt


# In[ ]:


# Targets identified from the original notebook
macs = {
    7.1: 'TkNDTnp',
    7.2: 'hPdqseP',
    7.3: 'QDPpKtd',
    8.1: 'DDPTprQq',
    8.2: 'rQpqRePQ',
    9.1: 'pPYhPKDLq',
    10.1: 'PEAARvpRlt',
    10.2: 'EvDPehpNap'
}

pdbs = {
    7.1: '6be9',
    7.2: '6bew',
    7.3: '6bf5',
    8.1: '6be7',
    8.2: '6ben',
    9.1: '6beo',
    10.1: '6beq',
    10.2: '6ber'
}


# In[ ]:


# Test a couple of targets (e.g., the heptapeptides 7.1 and 7.2)
# You can change this list to test all of them: list(macs.keys())
test_keys = [7.1]
results = {}

for k in test_keys:
    seq = macs[k]
    pdb_id = pdbs[k]
    print(f"\n{'='*50}\nOptimising {pdb_id} (Sequence: {seq})\n{'='*50}")
    
    t1 = time.time()
    
    # 1. Initialize the optimiser
    opt = CyclicPeptideOptimiser(seq)
    
    # 2. Build start mac using ISAMBARD's rand_mac
    print("Building starting macrocycle (this may take a moment for rand_mac to close the loop)...")
    opt.build_start_mac()
    
    # 3. Setup OpenMM
    print("Setting up OpenMM context...")
    opt.amber_setup()
    
    # 4. Run the fast optimisation loop
    # Using a small number of iterations (e.g. 50) for testing speed
    n_iter = 50
    samplesize = 40
    print(f"Running optimisation for {n_iter} iterations...")
    opt.optimise(
        n_iter=n_iter,
        samplesize=samplesize,
        wp_len=15, 
        hof_len=3, 
        rama_rmsd=5, 
        n_permute=5, 
        plot=False
    )
    
    t2 = time.time()
    print(f"\nFinished {pdb_id} in {t2 - t1:.2f} seconds.")
    print(f"Initial energy: {opt.energies[0]:.2f}")
    print(f"Best energy achieved: {opt.energies[-1]:.2f}")
    
    results[k] = opt
    
    # 5. Plot the energy trajectory
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(opt.energies)), opt.energies, marker='o', markersize=3)
    plt.title(f'Energy Trajectory for {pdb_id}')
    plt.xlabel('Iteration')
    plt.ylabel('Energy (kJ/mol)')
    plt.grid(True)
    plt.show()
    
    # 6. Save the best structure
    output_pdb = f'{pdb_id}_best.pdb'
    with open(output_pdb, 'w') as f:
        PDBFile.writeFile(opt.model.topology, opt.halloffame[0][1], file=f)
    print(f"Saved best structure to {output_pdb}")

