# Analyse de sensibilité : dpm_rpm, seuil 0.1, 0.15, 0.2, swc, clay
# ==> ne pas créer de dossier, faire une courbe 3D SOC(t=25, variable), faire un indicateur de sensibilité = dSOC/dvar(t, var)

"""

    Fichier d'entrée: 
    Un excel qui contient les colonnes :
    Year - 
    Month - 
    Temp - (month average) 
    Rain - (mm)
    ETP - monthly


"""


import pandas as pd
import numpy as np
from RothC import RothCModel_C
from datetime import datetime
import os


global depth, clay, dpm_rpm, iom, n_year
global data_path, input_file

#CONFIG
with open('config/config.txt', 'r') as file:
    config_content = file.read()

# Exécuter dynamiquement le contenu du fichier pour obtenir le dictionnaire
config_dict = eval(config_content)

# Extraire les valeurs et les assigner à des variables
HOME = config_dict['HOME']
input_data = config_dict['input_data']
output_data = config_dict['output_data']
excel_file = config_dict['excel_file']
excel_sheet = config_dict['excel_sheet']
output_folder = HOME + output_data + config_dict['output_folder'] 
n_year = int(config_dict['nbr_year'])
INIT_MODE = config_dict['init_mode']
swc = float(config_dict['swc'])
SHOW = bool(config_dict['SHOW'])
rmf_min = float(config_dict['rmf_min'])
decomp_params = config_dict['decomp_params']
decomp_ratio = config_dict['decomp_ratio']
fym_ratio = config_dict['fym_ratio']


if os.path.exists(output_folder):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f'{output_folder}_{current_time}'
    print(f"Folder already exists. Renaming to: {output_folder}")

os.makedirs(output_folder)

output_folder += '/'
output_name = "results_RothC_26_3_v2_C"
output_yearly_name = output_folder + output_name + "_year"
output_monthly_name = output_folder + output_name + "_month"
input_file = HOME + input_data + excel_file
data_path = HOME + output_data + 'data.csv'


#Zimmermann et al. (2007) proposed a soil fractionation scheme to obtain the initial C fractions in a soil and run the RothC model.
# Weihermuller et al. (2013),  Pedotransfer fucntion
#PEDOTRANSFER FUNCTION

if INIT_MODE == 'set':
    init_set_soc =list(config_dict['init_set_soc'])
else:
    init_set_soc = None

mode = 'RothC_26.3'
#mode = 'RothC_10_N'

from time import time
t=time()
model = RothCModel_C(
    time_fact=12, 
    excel_file= input_file, 
    excel_sheet= excel_sheet, 
    output_yearly_name=output_yearly_name, 
    output_monthly_name=output_monthly_name, 
    n_year = n_year,
    init_mode = INIT_MODE, 
    mode = mode,
    init_set_soc = init_set_soc,
    output_folder = output_folder, 
    rmf_min = rmf_min,
    decomp_params=decomp_params, 
    decomp_ratio=decomp_ratio, 
    fym_ratio = fym_ratio
) 
print("Time to Init:", time()-t)

t=time()
model.run()
print("Time to Run: ", time()-t)
print("Model ended with success")

import matplotlib.pyplot as plt
if SHOW == True :
    plt.show()



