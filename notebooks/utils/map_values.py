import pandas as pd
import sys
import pandas as pd
from pyensembl import EnsemblRelease


def map_to_res_nonres(value):
    if value == '01':
        return 'NONRES'
    elif value == '10':
        return 'RES'
    else:
        return '-'

def map_by_pam(value):
    if value == '1000':
        return 'TNBC'
    elif value == '0100':
        return 'LumA'
    elif value == '0010':
        return 'LumB'
    elif value == '0001':
        return 'HER2'
    else:
        return '0'

def map_by_ihc(value):
    if value == '000':
        return 'TNBC'
    
    elif value in ['100', '010', '110']:
        return 'LumA'
    
    elif value in ['001']:
        return 'HER2'
    
    elif value in ['111', '101', '011']:
        return 'LumB'
    
    else:
        return '0'

def map_to_int(value):
    if value == 'TNBC':
        return 0
    elif value == 'LumA':
        return 1
    elif value == 'LumB':
        return 2
    elif value == 'HER2':
        return 3
    else:
        return None

def gene_name_to_ids(name):
    data = EnsemblRelease(75)

    return data.gene_ids_of_gene_name(name)

def gene_id_to_name(id):
    data = EnsemblRelease(75)

    return data.gene_name_of_gene_id(id)