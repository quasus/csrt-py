import os.path
import pickle
from .feature import *


def _load_table(table_name):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return pickle.load(open(os.path.join(dir_path,
                                         "lookup_tables",
                                         table_name+".pkl"), "rb"))

# Load the tables we actually need
for table_name in ['CNnorm']:
    TableFeature.tables[table_name] = _load_table(table_name)
