def get_label(species, density):
    """
    :param species: string of species name. Should be "macro", "nereo", or "mixed"
    :param density: string of density. Should be "low" or "high"
    :return: int label

    >>> get_label("macro", "low")
    0
    >>> get_label("macro", "high")
    1
    >>> get_label("nereo", "low")
    2
    >>> get_label("nereo", "high")
    3
    >>> get_label("mixed", "low")
    4
    >>> get_label("mixed", "high")
    5
    >>> get_label("Macro", "High")
    1
    >>> get_label("neReO", "hIGh")
    3
    >>> get_label("MIXED", "HIGH")
    5
    """
    density = str(density).lower()
    species = str(species).lower()

    if density == 'low':
        den = 0
    elif density == 'high':
        den = 1
    else:
        raise RuntimeError(f"Density {density} does not have a defined label")

    if species == 'macro':
        spe = 0
    elif species == 'nereo':
        spe = 1
    elif species == 'mixed':
        spe = 2
    else:
        raise RuntimeError(f"Species {species} does not have a defined label")

    return (spe << 1) | den


def get_species_from_label(label):
    """
    :param label: int label
    :return: str kelp species

    >>> [get_species_from_label(i) for i in range(6)]
    ['macro', 'macro', 'nereo', 'nereo', 'mixed', 'mixed']
    """
    la = label >> 1
    if la == 0:
        return 'macro'
    elif la == 1:
        return 'nereo'
    elif la == 2:
        return "mixed"
    else:
        raise RuntimeError(f"Species for label {label} does not exist")


def get_density_from_label(label):
    """
    :param label: int label
    :return: str kelp density

    >>> [get_density_from_label(i) for i in range(6)]
    ['low', 'high', 'low', 'high', 'low', 'high']
    """
    la = label % 2
    if la == 0:
        return 'low'
    elif la == 1:
        return 'high'


if __name__ == "__main__":
    import doctest
    doctest.testmod()
