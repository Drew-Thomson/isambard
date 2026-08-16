from .non_canonical import convert_pro_to_hyp
from .scwrl import pack_side_chains_scwrl
from .daspr import pack_side_chains_daspr


def pack_side_chains(assembly, sequences, method='daspr', **kwargs):
    """Unified interface for side-chain packing.

    Parameters
    ----------
    assembly : AMPAL Assembly
    sequences : [str]
    method : str, optional
        'daspr' (default) or 'scwrl'.
    **kwargs :
        Passed to the underlying packing function (e.g., rigid_rotamer_model for scwrl).
    """
    if method == 'daspr':
        return pack_side_chains_daspr(assembly, sequences)
    elif method == 'scwrl':
        return pack_side_chains_scwrl(assembly, sequences, **kwargs)
    else:
        raise ValueError("Method must be 'daspr' or 'scwrl'.")
