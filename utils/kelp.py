def get_species_label(species):
    """
    :param species: string of species name. Should be "macro", "nereo", or "mixed"
    :param density: string of density. Should be "low" or "high"
    :return: int label

    >>> get_species_label("macro")
    1
    >>> get_species_label("nereo")
    2
    >>> get_species_label("mixed")
    3
    >>> get_species_label("Macro")
    1
    >>> get_species_label("neReO")
    2
    >>> get_species_label("MIXED")
    3
    """
    species = str(species).lower()

    try:
        result = ['macro', 'nereo', 'mixed'].index(species) + 1
    except ValueError:
        print(f"Species {species} does not have a defined label")
        # raise ValueError(f"Species {species} does not have a defined label")
        result = 100

    return result


def get_species_from_species_label(label):
    """
    :param label: int label
    :return: str kelp species

    >>> [get_species_from_species_label(i) for i in range(1, 4)]
    ['macro', 'nereo', 'mixed']
    """

    try:
        result = ['macro', 'nereo', 'mixed'][label-1]
    except IndexError:
        raise IndexError(f"Species for label {label} does not exist")

    return result


def get_species_density_label(species, density):
    """
    :param species: string of species name. Should be "macro", "nereo", or "mixed"
    :param density: string of density. Should be "low" or "high"
    :return: int label

    >>> get_species_density_label("macro", "low")
    0
    >>> get_species_density_label("macro", "high")
    1
    >>> get_species_density_label("nereo", "low")
    2
    >>> get_species_density_label("nereo", "high")
    3
    >>> get_species_density_label("mixed", "low")
    4
    >>> get_species_density_label("mixed", "high")
    5
    >>> get_species_density_label("Macro", "High")
    1
    >>> get_species_density_label("neReO", "hIGh")
    3
    >>> get_species_density_label("MIXED", "HIGH")
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


def get_species_from_species_density_label(label):
    """
    :param label: int label
    :return: str kelp species

    >>> [get_species_from_species_density_label(i) for i in range(6)]
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


def get_density_from_species_density_label(label):
    """
    :param label: int label
    :return: str kelp density

    >>> [get_density_from_species_density_label(i) for i in range(6)]
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
