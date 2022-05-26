from .base_class import Identity, Compressor
from .random_sparsification import RandomSparsifier

__all__ = ["get_compression", "Identity", "RandomSparsifier", "Compressor"]

def get_compression(comp_name):
    if comp_name == 'none':
        return Identity()
    elif comp_name == 'random_sparsification':
        return RandomSparsifier()
    else:
        raise ValueError(f"Unknown compression: comp_name")