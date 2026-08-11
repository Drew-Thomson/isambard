"""This module provides an interface to the program dASPR.

The dASPR executable must be on your path.
"""

import os
import subprocess
import tempfile
import ampal


def daspr_available():
    """True if dASPR is available."""
    available = False
    try:
        subprocess.check_output(['dASPR'], stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        available = True
    except FileNotFoundError:
        print("dASPR has not been found on your path.")
    return available


def run_daspr(pdb, sequence):
    """Runs dASPR on input PDB string and sequence string.

    Parameters
    ----------
    pdb : str
        PDB string.
    sequence : str
        Amino acid sequence for dASPR to pack in single-letter code.

    Returns
    -------
    daspr_pdb : str
        String of packed dASPR PDB.
    """
    pdb = pdb.encode()
    sequence = sequence.encode()
    try:
        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as daspr_tmp,\
                tempfile.NamedTemporaryFile(mode='w+b', delete=False) as daspr_seq,\
                tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as daspr_out:
            daspr_tmp.write(pdb)
            daspr_tmp.flush()
            daspr_seq.write(sequence)
            daspr_seq.flush()
            daspr_command = ['dASPR',
                             '-i', daspr_tmp.name,
                             '-o', daspr_out.name,
                             '-s', daspr_seq.name]
            subprocess.check_output(daspr_command)
            daspr_out.seek(0)
            daspr_pdb = daspr_out.read()
    finally:
        os.remove(daspr_tmp.name)
        os.remove(daspr_out.name)
        os.remove(daspr_seq.name)
    return daspr_pdb.decode()


def pack_side_chains_daspr(assembly, sequences):
    """Packs side chains onto a protein structure using dASPR.

    Parameters
    ----------
    assembly : AMPAL Assembly
        AMPAL object containing some protein structure.
    sequences : [str]
        A list of amino acid sequences in single-letter code for dASPR to pack.

    Returns
    -------
    packed_structure : AMPAL Assembly
        A new AMPAL Assembly containing the packed structure.
    """
    if not daspr_available():
        raise ValueError('dASPR is unavailable on your system path.')
    protein = [x for x in assembly if isinstance(x, ampal.Polypeptide)]
    total_seq_len = sum([len(x) for x in sequences])
    total_aa_len = sum([len(x) for x in protein])
    if total_seq_len != total_aa_len:
        raise ValueError('Total sequence length ({}) does not match '
                         'total Polypeptide length ({}).'.format(
                             total_seq_len, total_aa_len))
    if len(protein) != len(sequences):
        raise ValueError('Number of sequences ({}) does not match '
                         'number of Polypeptides ({}).'.format(
                             len(sequences), len(protein)))
    
    packed_pdb = run_daspr(assembly.pdb, ''.join(sequences))
    
    new_assembly = ampal.load_pdb(packed_pdb, path=False)
    return new_assembly


__author__ = 'Christopher W. Wood'
