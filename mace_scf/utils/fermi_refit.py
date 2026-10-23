"""The former independent EF-refit entry point is retired."""


def refit_fermi_readout(*args, **kwargs):
    raise ValueError('Independent EF refitting has been removed. Train energy, '
                     'forces and EF jointly; no observer-only offset or refit '
                     'may change the energy-derived voltage.')
