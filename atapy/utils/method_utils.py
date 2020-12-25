import sys
from pathlib import Path
from typing import Dict
import pandas as pd

""" Common functions related to type checking/class loading and other general handling in methods"""

def load_classes_in_dir(dir_path: Path) -> None:
    """ Loads all python classes in a given directory (in filenames ending with .py) """
    for py_file_name in [f.name for f in dir_path.iterdir()
                         if f.suffix == '.py' and f.name != '__init__.py']:
        # For the class_path we should only include the part of the path from classes and forward
        parts = dir_path.parts
        index = max(loc for loc, val in enumerate(parts) if val == 'atapy')
        mod = __import__('.'.join(parts[index:]), fromlist=[py_file_name])
        classes = [getattr(mod, x) for x in dir(mod) if isinstance(getattr(mod, x), type)]
        for sub_cls in classes:
            setattr(sys.modules[__name__], sub_cls.__name__, sub_cls)


def to_list(var):
    """ Create a list of an item if it wasn't already"""
    if isinstance(var, list):
        return var
    else:
        return [var]


def check_types(data_df: pd.DataFrame, correct_types: Dict):
    """ Checks that the given data_df contains the right columns and data types. If not it tries to convert them """
    if set(data_df.columns) != set(correct_types.keys()):
        raise ValueError("Data provided with incorrect columns. Expecting {}".format(
            correct_types.keys()))
    if data_df.dtypes.to_dict() != correct_types:
        data_df = data_df.astype(correct_types)
    return data_df
