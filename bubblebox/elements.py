from json import dump, loads
import numpy as np
from ast import literal_eval


elements_database_json = """
{
    "H": {
        "name": "Hydrogen",
        "atomic_number": 1
    },
    "He": {
        "name": "Helium",
        "atomic_number": 2
    },
    "Li": {
        "name": "Lithium",
        "atomic_number": 3
    },
    "Be": {
        "name": "Beryllium",
        "atomic_number": 4
    },
    "B": {
        "name": "Boron",
        "atomic_number": 5
    },
    "C": {
        "name": "Carbon",
        "atomic_number": 6
    },
    "N": {
        "name": "Nitrogen",
        "atomic_number": 7
    },
    "O": {
        "name": "Oxygen",
        "atomic_number": 8
    },
    "F": {
        "name": "Fluorine",
        "atomic_number": 9
    },
    "Ne": {
        "name": "Neon",
        "atomic_number": 10
    },
    "Na": {
        "name": "Sodium",
        "atomic_number": 11
    },
    "Mg": {
        "name": "Magnesium",
        "atomic_number": 12
    },
    "Al": {
        "name": "Aluminium",
        "atomic_number": 13
    },
    "Si": {
        "name": "Silicon",
        "atomic_number": 14
    },
    "P": {
        "name": "Phosphorus",
        "atomic_number": 15
    },
    "S": {
        "name": "Sulfur",
        "atomic_number": 16
    },
    "Cl": {
        "name": "Chlorine",
        "atomic_number": 17
    },
    "Ar": {
        "name": "Argon",
        "atomic_number": 18
    },
    "K": {
        "name": "Potassium",
        "atomic_number": 19
    },
    "Ca": {
        "name": "Calcium",
        "atomic_number": 20
    },
    "Sc": {
        "name": "Scandium",
        "atomic_number": 21
    },
    "Ti": {
        "name": "Titanium",
        "atomic_number": 22
    },
    "V": {
        "name": "Vanadium",
        "atomic_number": 23
    },
    "Cr": {
        "name": "Chromium",
        "atomic_number": 24
    },
    "Mn": {
        "name": "Manganese",
        "atomic_number": 25
    },
    "Fe": {
        "name": "Iron",
        "atomic_number": 26
    },
    "Co": {
        "name": "Cobalt",
        "atomic_number": 27
    },
    "Ni": {
        "name": "Nickel",
        "atomic_number": 28
    },
    "Cu": {
        "name": "Copper",
        "atomic_number": 29
    },
    "Zn": {
        "name": "Zinc",
        "atomic_number": 30
    },
    "Ga": {
        "name": "Gallium",
        "atomic_number": 31
    },
    "Ge": {
        "name": "Germanium",
        "atomic_number": 32
    },
    "As": {
        "name": "Arsenic",
        "atomic_number": 33
    },
    "Se": {
        "name": "Selenium",
        "atomic_number": 34
    },
    "Br": {
        "name": "Bromine",
        "atomic_number": 35
    },
    "Kr": {
        "name": "Krypton",
        "atomic_number": 36
    },
    "Rb": {
        "name": "Rubidium",
        "atomic_number": 37
    },
    "Sr": {
        "name": "Strontium",
        "atomic_number": 38
    },
    "Y": {
        "name": "Yttrium",
        "atomic_number": 39
    },
    "Zr": {
        "name": "Zirconium",
        "atomic_number": 40
    },
    "Nb": {
        "name": "Niobium",
        "atomic_number": 41
    },
    "Mo": {
        "name": "Molybdenum",
        "atomic_number": 42
    },
    "Tc": {
        "name": "Technetium",
        "atomic_number": 43
    },
    "Ru": {
        "name": "Ruthenium",
        "atomic_number": 44
    },
    "Rh": {
        "name": "Rhodium",
        "atomic_number": 45
    },
    "Pd": {
        "name": "Palladium",
        "atomic_number": 46
    },
    "Ag": {
        "name": "Silver",
        "atomic_number": 47
    },
    "Cd": {
        "name": "Cadmium",
        "atomic_number": 48
    },
    "In": {
        "name": "Indium",
        "atomic_number": 49
    },
    "Sn": {
        "name": "Tin",
        "atomic_number": 50
    },
    "Sb": {
        "name": "Antimony",
        "atomic_number": 51
    },
    "Te": {
        "name": "Tellurium",
        "atomic_number": 52
    },
    "I": {
        "name": "Iodine",
        "atomic_number": 53
    },
    "Xe": {
        "name": "Xenon",
        "atomic_number": 54
    },
    "Cs": {
        "name": "Cesium",
        "atomic_number": 55
    },
    "Ba": {
        "name": "Barium",
        "atomic_number": 56
    },
    "La": {
        "name": "Lanthanum",
        "atomic_number": 57
    },
    "Ce": {
        "name": "Cerium",
        "atomic_number": 58
    },
    "Pr": {
        "name": "Praseodymium",
        "atomic_number": 59
    },
    "Nd": {
        "name": "Neodymium",
        "atomic_number": 60
    },
    "Pm": {
        "name": "Promethium",
        "atomic_number": 61
    },
    "Sm": {
        "name": "Samarium",
        "atomic_number": 62
    },
    "Eu": {
        "name": "Europium",
        "atomic_number": 63
    },
    "Gd": {
        "name": "Gadolinium",
        "atomic_number": 64
    },
    "Tb": {
        "name": "Terbium",
        "atomic_number": 65
    },
    "Dy": {
        "name": "Dysprosium",
        "atomic_number": 66
    },
    "Ho": {
        "name": "Holmium",
        "atomic_number": 67
    },
    "Er": {
        "name": "Erbium",
        "atomic_number": 68
    },
    "Tm": {
        "name": "Thulium",
        "atomic_number": 69
    },
    "Yb": {
        "name": "Ytterbium",
        "atomic_number": 70
    },
    "Lu": {
        "name": "Lutetium",
        "atomic_number": 71
    },
    "Hf": {
        "name": "Hafnium",
        "atomic_number": 72
    },
    "Ta": {
        "name": "Tantalum",
        "atomic_number": 73
    },
    "W": {
        "name": "Tungsten",
        "atomic_number": 74
    },
    "Re": {
        "name": "Rhenium",
        "atomic_number": 75
    },
    "Os": {
        "name": "Osmium",
        "atomic_number": 76
    },
    "Ir": {
        "name": "Iridium",
        "atomic_number": 77
    },
    "Pt": {
        "name": "Platinum",
        "atomic_number": 78
    },
    "Au": {
        "name": "Gold",
        "atomic_number": 79
    },
    "Hg": {
        "name": "Mercury",
        "atomic_number": 80
    },
    "Tl": {
        "name": "Thallium",
        "atomic_number": 81
    },
    "Pb": {
        "name": "Lead",
        "atomic_number": 82
    },
    "Bi": {
        "name": "Bismuth",
        "atomic_number": 83
    },
    "Po": {
        "name": "Polonium",
        "atomic_number": 84
    },
    "At": {
        "name": "Astatine",
        "atomic_number": 85
    },
    "Rn": {
        "name": "Radon",
        "atomic_number": 86
    },
    "Fr": {
        "name": "Francium",
        "atomic_number": 87
    },
    "Ra": {
        "name": "Radium",
        "atomic_number": 88
    },
    "Ac": {
        "name": "Actinium",
        "atomic_number": 89
    },
    "Th": {
        "name": "Thorium",
        "atomic_number": 90
    },
    "Pa": {
        "name": "Protactinium",
        "atomic_number": 91
    },
    "U": {
        "name": "Uranium",
        "atomic_number": 92
    },
    "Np": {
        "name": "Neptunium",
        "atomic_number": 93
    },
    "Pu": {
        "name": "Plutonium",
        "atomic_number": 94
    },
    "Am": {
        "name": "Americium",
        "atomic_number": 95
    },
    "Cm": {
        "name": "Curium",
        "atomic_number": 96
    },
    "Bk": {
        "name": "Berkelium",
        "atomic_number": 97
    },
    "Cf": {
        "name": "Californium",
        "atomic_number": 98
    },
    "Es": {
        "name": "Einsteinium",
        "atomic_number": 99
    },
    "Fm": {
        "name": "Fermium",
        "atomic_number": 100
    },
    "Md": {
        "name": "Mendelevium",
        "atomic_number": 101
    },
    "No": {
        "name": "Nobelium",
        "atomic_number": 102
    },
    "Lr": {
        "name": "Lawrencium",
        "atomic_number": 103
    },
    "Rf": {
        "name": "Rutherfordium",
        "atomic_number": 104
    },
    "Db": {
        "name": "Dubnium",
        "atomic_number": 105
    },
    "Sg": {
        "name": "Seaborgium",
        "atomic_number": 106
    },
    "Bh": {
        "name": "Bohrium",
        "atomic_number": 107
    },
    "Hs": {
        "name": "Hassium",
        "atomic_number": 108
    },
    "Mt": {
        "name": "Meitnerium",
        "atomic_number": 109
    },
    "Ds": {
        "name": "Darmstadtium",
        "atomic_number": 110
    },
    "Rg": {
        "name": "Roentgenium",
        "atomic_number": 111
    },
    "Cn": {
        "name": "Copernicium",
        "atomic_number": 112
    },
    "Nh": {
        "name": "Nihonium",
        "atomic_number": 113
    },
    "Fl": {
        "name": "Flerovium",
        "atomic_number": 114
    },
    "Mc": {
        "name": "Moscovium",
        "atomic_number": 115
    },
    "Lv": {
        "name": "Livermorium",
        "atomic_number": 116
    },
    "Ts": {
        "name": "Tennessine",
        "atomic_number": 117
    },
    "Og": {
        "name": "Oganesson",
        "atomic_number": 118
    }
}
"""

element_database = loads(elements_database_json)

def get_atomic_number(s):
    """
    Returns the atomic number of the element s (string)

    Example usage:
    an = get_atmic_number("He") # returns 2
    """

    return element_database[s]["atomic_number"]

def get_full_name(s):
    """
    Returns the full name of the element s (string)

    Example usage:
    name = get_full_name("He") # returns "Helium"
    """

    return element_database[s]["name"]



