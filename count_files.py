import os
from pathlib import Path

def print_count_in_file(path, ext="*.png"):
    path = Path(path)

    for sub in ["x", "y"]:
        print(path.joinpath(sub))
        print(len(list(path.joinpath(sub).glob(ext))))

if __name__ == '__main__':
    print_count_in_file("data/datasets/Calvert_2012")
    print_count_in_file("data/datasets/Calvert_2015")
    print_count_in_file("data/datasets/Calvert_WestBeach_2016")
    print_count_in_file("data/datasets/Calvert_ChokedNorthBeach_2016")