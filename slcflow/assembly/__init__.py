"""State packing, frozen inputs, and residual assembly (ARCH-3.2/3.3,
ARCH-5.1; Theory Manual sections 5.3-5.4, 6.1, A.7).

The BC-switched (BackPressureSpec) residual form is deferred to M5 per
ARCH-8; ``FrozenInputs`` rejects it until then.
"""
from .assembler import AssembledFields, ResidualAssembler
from .inputs import ClosureFields, FrozenInputs
from .pack import n_unknowns, pack, unpack

__all__ = [
    "AssembledFields",
    "ClosureFields",
    "FrozenInputs",
    "ResidualAssembler",
    "n_unknowns",
    "pack",
    "unpack",
]
