# Analyse de sensibilité : dpm_rpm, seuil 0.1, 0.15, 0.2, swc, clay
# ==> ne pas créer de dossier, faire une courbe 3D SOC(t=25, variable), faire un indicateur de sensibilité = dSOC/dvar(t, var)


#bilan ==> tres peu de sensibilite au parametre clay et swc initiale mais sensibilité forte au paramètre rmf_min
import pandas as pd
import numpy as np
from RothC import RothCModel_C
from extraire_format_roth_excel import *
from datetime import datetime
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from SALib.sample import saltelli, fast_sampler
from SALib.analyze import sobol, fast
import numpy as np


#ajouter dpm_rpm pour analyse de sensibilité

global n_year, depth
global data_path, input_file, init_soil_carbon
global step 
global df
step = 10

# Create a DataFrame
df = pd.DataFrame()



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
swc = int(config_dict['swc'])
SHOW = bool(config_dict['SHOW'])


output_folder += '/'
output_name = "results_RothC_26_3_v2_C"
output_years_path = output_folder + output_name + "_year"
output_months_path = output_folder + output_name + "_month"
input_file = HOME + input_data + excel_file
data_path = HOME + output_data + 'data.csv'



def show_plots(var_list, range_list):
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # Add more colors if needed
    blue_values = np.linspace(0, 1, step)  # Création d'une séquence de valeurs bleues de 0 à 1

    for i in range(step):
        y1 = range(len(var_list[0]))
        x1 = [range_list[i]] * len(var_list[0])
        z1 = var_list[i]
        #ax.plot(x1, y1, z1, label='Courbe 1', color=colors[i%len(colors)])
        color = (0, 0, blue_values[i])  # Utilisation de valeurs RGB avec bleu variant
        ax.plot(x1, y1, z1, label=f'Courbe {i+1}', color=color)
    
    plt.show()
    plt.plot(range_list, [row[-1] for row in var_list])
    plt.title('Sensibilidad del modelo RothC al parametro x')
    plt.xlabel('x')
    plt.ylabel('SOC t/ha/ano')
    plt.show()


def cumul_CO2(soc_list, x_range, name):
    #show_plots(soc_list, x_range)
    soc_cumul = [sum(soc_list[i])/len(soc_list[i]) for i in range(len(x_range))]

    df[f'{name}_range'] = x_range
    df[f'{name}_soc_cumul'] = soc_cumul


def sensibilite_dpm_rpm(dpm_min, dpm_max):

    dpm_rpm_range = np.linspace(dpm_rpm_min, dpm_rpm_max, step)    
    soc_list = [0] * step
    clay, depth, trash, init_soil_carbon = extraire_data(input_file, data_path, excel_sheet)
    for i in range(step):
        df = pd.read_csv(data_path)
        df['DPM_RPM'] = dpm_rpm_range[i]
        df.to_csv(data_path, index=False)

        model = RothCModel_C(
            time_fact=12, 
            soc = init_soil_carbon,
            clay=clay, 
            depth=depth, 
            swc=swc, 
            data_path=data_path, 
            output_years_path=output_years_path, 
            output_months_path=output_months_path, 
            n_year = n_year,
            output_folder = output_folder, 
            dev_mode = True
        ) 
            
        soc_list[i] = model.run()
    cumul_CO2(soc_list, dpm_rpm_range, 'dpm_rpm')

def sensibilite_clay(clay_min, clay_max):
    
    clay_range = np.linspace(clay_min, clay_max, step)
    soc_list = [0] * step
    clay, depth, trash, init_soil_carbon = extraire_data(input_file, data_path, excel_sheet)
    for i in range(step):
        clay = clay_range[i]
        model = RothCModel_C(
            time_fact=12, 
            soc = init_soil_carbon,
            clay=clay, 
            depth=depth, 
            swc=swc, 
            data_path=data_path, 
            output_years_path=output_years_path, 
            output_months_path=output_months_path, 
            n_year = n_year,
            output_folder = output_folder, 
            dev_mode = True
        ) 
            
        soc_list[i] = model.run()
    cumul_CO2(soc_list, clay_range, 'clay')

def sensibilite_swc(swc_min, swc_max):
    swc_range = np.linspace(swc_min, swc_max, step)

    soc_list = [0] * step
    clay, depth, trash, init_soil_carbon = extraire_data(input_file, data_path, excel_sheet)

    for i in range(step):
        
        model = RothCModel_C(
            time_fact=12, 
            soc = init_soil_carbon,
            clay=clay, 
            depth=depth, 
            swc=swc_range[i], 
            data_path=data_path, 
            output_years_path=output_years_path, 
            output_months_path=output_months_path, 
            n_year = n_year,
            output_folder = output_folder, 
            dev_mode = True
        ) 
            
        soc_list[i] = model.run()
    cumul_CO2(soc_list, swc_range, 'swc')

def sensibilite_rmf_min(rmf_min, rmf_max):
    rmf_range = np.linspace(rmf_min, rmf_max, step)

    soc_list = [0] * step
    clay, depth, trash, init_soil_carbon = extraire_data(input_file, data_path, excel_sheet)

    for i in range(step):
        
        model = RothCModel_C(
            time_fact=12, 
            soc = init_soil_carbon,
            clay=clay, 
            depth=depth, 
            swc=swc, 
            data_path=data_path, 
            output_years_path=output_years_path, 
            output_months_path=output_months_path, 
            n_year = n_year,
            output_folder = output_folder, 
            dev_mode = True,
            rmf_min = rmf_range[i],
        ) 
            
        soc_list[i] = model.run()
    cumul_CO2(soc_list, rmf_range, 'rmf_min')

def sensibilite_dpm(dpm_k_min, dpm_k_max):
    dpm_k_range = np.linspace(k_dpm_min, k_dpm_max, step)

    soc_list = [0] * step
    clay, depth, trash, init_soil_carbon = extraire_data(input_file, data_path, excel_sheet)

    for i in range(step):
        
        model = RothCModel_C(
            time_fact=12, 
            soc = init_soil_carbon,
            clay=clay, 
            depth=depth, 
            swc=swc, 
            data_path=data_path, 
            output_years_path=output_years_path, 
            output_months_path=output_months_path, 
            n_year = n_year,
            output_folder = output_folder, 
            dev_mode = True,
            decomp_params = [dpm_k_range[i], 0.3, 0.66, 0.02]
        ) 
            
        soc_list[i] = model.run()
    cumul_CO2(soc_list, dpm_k_range, 'k_dpm')

def sensibilite_rpm(rpm_k_min, rpm_k_max):
    rpm_k_range = np.linspace(k_rpm_min, k_rpm_max, step)

    soc_list = [0] * step
    clay, depth, trash, init_soil_carbon = extraire_data(input_file, data_path, excel_sheet)

    for i in range(step):
        
        model = RothCModel_C(
            time_fact=12, 
            soc = init_soil_carbon,
            clay=clay, 
            depth=depth, 
            swc=swc, 
            data_path=data_path, 
            output_years_path=output_years_path, 
            output_months_path=output_months_path, 
            n_year = n_year,
            output_folder = output_folder, 
            dev_mode = True,
            decomp_params = [10, rpm_k_range[i], 0.66, 0.02]
        ) 
            
        soc_list[i] = model.run()
    cumul_CO2(soc_list, rpm_k_range, 'k_rpm')

def sensibilite_soc(soc_min, soc_max):
    soc_range = np.linspace(soc_min, soc_max, step)
    soc_list = [0] * step
    clay, depth, trash, soc = extraire_data(input_file, data_path, excel_sheet)

    for i in range(step):
        
        model = RothCModel_C(
            time_fact=12, 
            clay=clay, 
            depth=depth, 
            soc=soc_range[i],
            swc=swc, 
            data_path=data_path, 
            output_years_path=output_years_path, 
            output_months_path=output_months_path, 
            n_year = n_year,
            output_folder = output_folder, 
            dev_mode = True,
        ) 
            
        soc_list[i] = model.run()
    cumul_CO2(soc_list, soc_range, 'soc')


def sensibilite_pools(combinations):
    pools=[0]*len(combinations)
    clay, depth, trash, soc = extraire_data(input_file, data_path, excel_sheet)
    print(type(soc))
    for i  in range(len(combinations)):
        comb=list(combinations[i])
        iom=0.049 * soc**1.139
        init_soil_carbon = [ j*(soc-iom) for j in comb]
        init_soil_carbon.append(iom)

        
        model = RothCModel_C(
            time_fact=12, 
            clay=clay, 
            depth=depth, 
            soc=init_soil_carbon,
            swc=swc, 
            data_path=data_path, 
            output_years_path=output_years_path, 
            output_months_path=output_months_path, 
            n_year = n_year,
            output_folder = output_folder, 
            dev_mode = True,
        ) 
            
        pools[i] = model.run()
    cumul_CO2(pools, range(len(combinations)), 'pools')

dpm_rpm_min = 0.01
dpm_rpm_max = 0.7

clay_min = 10
clay_max = 50

swc_min = -40
swc_max = 0

rmf_min = 0.10
rmf_max = 0.20

k_dpm_min = 1
k_dpm_max = 10

k_rpm_min = 0.05
k_rpm_max = 0.7

soc_min=35
soc_max=60


n_div = 6

from itertools import product

def find_combinations(n):
    # Générer l'ensemble des valeurs possibles
    values = [i / n for i in range(n)]
    
    # Trouver toutes les combinaisons de 4 valeurs
    all_combinations = product(values, repeat=4)
    
    # Filtrer les combinaisons pour ne garder que celles dont la somme est égale à 1
    valid_combinations = [comb for comb in all_combinations if sum(comb) == 1]
    
    return valid_combinations

# Exemple d'utilisation
combinations = find_combinations(n_div)

file_path = 'output4.xlsx'


"""
sensibilite_dpm_rpm(dpm_rpm_min, dpm_rpm_max)
sensibilite_clay(clay_min, clay_max)
sensibilite_swc(swc_min, swc_max)
sensibilite_rmf_min(rmf_min, rmf_max)
sensibilite_dpm(k_dpm_min, k_dpm_max)
sensibilite_rpm(k_rpm_min, k_rpm_max)
sensibilite_soc(soc_min, soc_max)
"""

sensibilite_pools(combinations)
# Save the DataFrame to an Excel file
df.to_excel(file_path, index=False)
print(f"Excel file created at: {file_path}")








#***************************************************************************************************************

def sensibility_model(rmf_min, dpm_rpm, clay, soc):
    print(f'model RothC init with params clay = {clay}, rmf_min = {rmf_min}, dpm_rpm = {dpm_rpm}, swc = {swc}')
    model = RothCModel_C(
        time_fact=12, 
        soc = soc,
        clay=clay, 
        depth=depth, 
        swc=swc, 
        data_path=data_path, 
        output_years_path=output_years_path, 
        output_months_path=output_months_path, 
        n_year = n_year,
        output_folder = output_folder, 
        dev_mode = True, 
        sensibility_params = [rmf_min, dpm_rpm, 1],
    ) 
    plt.close('all')
    return model.run()



clay, depth, trash, init_soil_carbon = extraire_data(input_file, data_path, excel_sheet)


# Define the ranges for the parameters
dpm_rpm_min = 0.04
dpm_rpm_max = 0.50

rmf_min = 0.10
rmf_max = 0.20

k_dpm_min = 8
k_dpm_max = 12

k_rpm_min = 0.05
k_rpm_max = 0.7

bare = 0
cover = 1


# Define the problem for SALib
problem = {
    'num_vars': 4,
    'names': ['rmf_min', 'dpm_rpm', 'clay', 'soc'],
    'bounds': [
        [rmf_min, rmf_max],
        [dpm_rpm_min, dpm_rpm_max],
        [clay_min, clay_max],
        [soc_min, soc_max]
    ]
}


def sobol_analysis():
    # Generate parameter samples
    param_values = saltelli.sample(problem, 30)

    print('taille params: ', len(param_values))

    outputs = np.array([sensibility_model(x[0], x[1], x[2], x[3]) for x in param_values])

    print(outputs)

    # Perform Sobol sensitivity analysis for each time step
    sobol_indices = []
    for t in range(outputs.shape[1]):  # Assuming outputs is of shape (n_samples, time_steps)
        Si = sobol.analyze(problem, outputs[:, t])
        sobol_indices.append(Si)




    print(sobol_indices)
    plt.close('all')
    # Example: Plot the first-order Sobol indices over time for each parameter
    time_steps = range(outputs.shape[1])
    for i, name in enumerate(problem['names']):
        first_order = [Si['S1'][i] for Si in sobol_indices]
        plt.plot(time_steps, first_order, label=name)

    plt.xlabel('Time Step')
    plt.ylabel('First-order Sobol Index')
    plt.legend()
    plt.title('Sensitivity Analysis Over Time')
    plt.show()

def fast_analysis():
    param_values_fast = fast_sampler.sample(problem, 65)
    outputs_fast = np.array([sensibility_model(x[0], x[1], x[2], x[3]) for x in param_values_fast])


    # Perform FAST sensitivity analysis for each time step
    fast_indices = []
    for t in range(outputs_fast.shape[1]):
        Si = fast.analyze(problem, outputs_fast[:, t], M=4)
        fast_indices.append(Si)

    # Example: Plot the first-order FAST indices over time for each parameter
    time_steps = range(outputs_fast.shape[1])
    plt.close('all')
    for i, name in enumerate(problem['names']):
        first_order_fast = [Si['S1'][i] for Si in fast_indices]
        plt.plot(time_steps, first_order_fast, label=f'FAST - {name}')

    plt.xlabel('Time Step')
    plt.ylabel('First-order FAST Index')
    plt.legend()
    plt.title('FAST Sensitivity Analysis Over Time')
    plt.show()

#sobol_analysis()
#fast_analysis()


"""


def sensibilite_general(x_min, x_max):
    x_range = np.linspace(x_min, x_max, step)
    soc_list = [0] * step
    clay, depth, trash, soc = extraire_data(input_file, data_path, excel_sheet)


    for i in range(step):

        model = RothCModel_C(
            time_fact=12, 
            clay= x['clay'][i], 
            depth = depth, 
            soc= x['soc'][i],
            swc= x['swc'][i], 
            data_path=data_path, 
            output_years_path=output_years_path, 
            output_months_path=output_months_path, 
            n_year = n_year,
            output_folder = output_folder, 
            dev_mode = True,
        ) 
            
        soc_list[i] = model.run()
        cumul_CO2(soc_list, x_range)
    return soc_list
"""