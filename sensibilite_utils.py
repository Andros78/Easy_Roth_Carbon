

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
    soc_cumul = [sum(soc_list[i]) for i in range(step)]

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

