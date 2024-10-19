"""
    Python version of The Rothamsted carbon model (RothC) 26.3.
    RothC is a model for the turnover of organic carbon in non-waterlogged topsoil that allows 
    for the effects of soil type, temperature, soil moisture and plant cover on the turnover process.

    Annex A Pedotransfer functions used to calculate the hydraulic properties from
    Farina et at 2013 - Modification of the RothC model for simulations of soil organic C dynamics in dryland regions
    https://ars.els-cdn.com/content/image/1-s2.0-S0016706113000438-mmc1.pdf


    M_field_capacity(0.05 bar)
    M_b(1 bar)
    M(15 bar) - wilting point!
    M_c(1000 bar)

"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pedotransfer_functions import *

"""
RothC:
afficher une image de la convergence de la descente
calibrer avec une base de donnée
pouvoir modifier plus de paramètres (rmf_min etc)
faire des vérifications sur les variables d'entrée dans config. Renvoyer des erreurs dans un fichier.txt ?
loop sur 12 premiers lignes si C_inp ou autre vide
publier sur GitHub, avec une documentation
faire un document qui explique les détails et fait une analyse de sensibilité du modèle. 

datasets potentiels: One hundred soil profiles were selected from Soria (2002) and Parra et al. (2003) and the annual carbon input for olive groves under conventional management was calculated from these data.

To run RothC to equilibrium (Coleman & Jenkinson, 1996), we needed to assume that the soils were in equilibrium (>30 yr with the same management)
"""

class RothCModel_C:
    def __init__(self, time_fact, excel_file, excel_sheet, output_folder, output_yearly_name, output_monthly_name, n_year, mode = 'RothC_26.3', init_mode = 'spin', init_set_soc = None,  rmf_min = 0.2, dev_mode = False, decomp_params=[10, 0.3, 0.66, 0.02], decomp_ratio=[0.46, 0.54], fym_ratio=[0.49, 0.49, 0.02]):
        """
        Parameters: 
        --> time_fact : step between two dates is 1/time_fact years. Usually is time_fact = 12 months.
        --> clay : % of clay in the soil. Is a number between 0 and 100
        --> depth : depth of the analyse soil in cm. per default set to 30
        --> init_mode :  init mode is to choose between Weihermuller, Spin and Set 
        --> mode : Choose between 'RothC_26.3' and 'RothC_10_N'. RothC_26.3 is the original Coleman model. RothC_10_N is the semi-arid/mediteranean modified model by Farina et al, 2013
        --> init_set_soc : When init_mode is 'Set', the program requires a list with initial pools values : [dpm, rpm, hum, bio, iom] 
        --> n_year : Number of years to simulate
        --> dev_mode: Boolean. If True, the output is not ploted and saved


        Input and output Path :
        --> excel_file : name of the excel file that should be stored in input_data folder
        --> excel_sheet : excel_file sheet name of the scenario
        --> output_folder : folder path of the output results
        --> output_yearly_name : name of the .xlsx yearly result
        --> output_monthly_name : name of the .xlsx monthly result

        Hyper-parameters: 
        --> rmf_min : default set to 0.2 
        --> decomp_params : default set to [10, 0.3, 0.66, 0.02]
        --> decomp_ratio : [0.46, 0.54]
        --> fym_ratio : [0.49, 0.49, 0.02]

        """
        

        self.time_fact = time_fact  
        self.init_mode = init_mode    
        self.init_set_soc = init_set_soc
        self.n_year = n_year
        self.swc = 0  # moisture deficit content of the soil
        self.rmf_min = rmf_min
        self.bulk_density = 1.3 # à modifier
        self.mode = mode



        self.output_monthly_name=output_monthly_name + '.xlsx'
        self.output_yearly_name=output_yearly_name + '.xlsx'
        self.output_folder = output_folder
        self.dev_mode = dev_mode
        self.error_journal = ""

        self.dpm_k, self.rpm_k, self.bio_k, self.hum_k  = decomp_params
        

        if sum(fym_ratio) !=1: 
            self.error_journal += "Fym ratio sum doesnt make 1, fym ratio set to default [0.49, 0.49, 0.02] \n"
            fym_ratio=[0.49, 0.49, 0.02]
        self.fym_dpm_ratio, self.fym_rpm_ratio, self.fym_hum_ratio = fym_ratio

        if sum(decomp_ratio) !=1: 
            self.error_journal += 'Decomp ratio sum doesnt make 1, decomp ratio set to default [0.46, 0.54]\n'
            decomp_ratio=[0.46, 0.54]
        self.decomp_ratio_bio, self.decomp_ratio_hum = decomp_ratio

        succeed = self.import_data(excel_file, excel_sheet)
        if succeed : 
            init_data = {
                'Parameter': ['Clay', 'Depth', 'SOC', 'DPM', 'RPM', 'HUM', 'BIO', 'IOM'],
                'Value': [self.clay, self.depth, self.soc, self.dpm, self.rpm, self.hum, self.bio, self.iom]
            }
            init_df = pd.DataFrame(init_data)
            self.error_journal += f"Model init with sucess:\n\n Parameters set to : \n{init_df}\n\n"
        else : 
            self.error_journal += "Model could not init\n"


        hyperparameters = {
            'Hyperparameters': ['dpm_k', 'rpm_k', 'bio_k', 'hum_k', 'fym_dpm', 'fym_rpm', 'fym_hum', 'decomp_bio', 'decomp_hum'],
            'Value': [self.dpm_k, self.rpm_k, self.bio_k, self.hum_k, self.fym_dpm_ratio, self.fym_rpm_ratio, self.fym_hum_ratio, self.decomp_ratio_bio, self.decomp_ratio_hum]
        }
        hyperparameters_df = pd.DataFrame(hyperparameters)
        self.error_journal += f"\nHyperparameters set to:\n {hyperparameters_df}\n\n"




    def import_data(self, excel_file, excel_sheet):
        """ Import the weather data and check if missing data"""

        self.df = pd.read_excel(excel_file, sheet_name=excel_sheet) 
        
        try: 
            clay_series = self.df[self.df.columns[self.df.columns.str.contains('Clay', case=False)]].iloc[0]
            self.clay = float(clay_series.iloc[0])

            #à modifier !
            self.silt = 10

            depth_series = self.df[self.df.columns[self.df.columns.str.contains('Depth', case=False)]].iloc[0]
            self.depth = float(depth_series.iloc[0])

            dpm_rpm_series = self.df[self.df.columns[self.df.columns.str.contains('DPM_RPM', case=False)]].iloc[0]
            self.dpm_rpm = float(dpm_rpm_series.iloc[0])

            soc_series = self.df[self.df.columns[self.df.columns.str.contains('SOC', case=False)]].iloc[0]
            self.soc = float(soc_series.iloc[0])

        except:
            self.clay, self.depth, self.dpm_rpm, self.soc = 0.27, 30, 1.44, 37
            self.error_journal += "ERROR: Clay, depth, dpm_rpm or soc are not in the input file. Default value clay = 0.27,  depth=30, soc = 37  have been set.\n"
        
        self.tsmd_max = -(20 + 1.3 * self.clay - 0.01 * (self.clay * self.clay)) * self.depth / 23.0
        self.oc = self.soc/(self.bulk_density*self.depth)
 

         
        self.details = {}
        self.details["year"] = self.df['Year'].tolist()
        self.details["month"] = self.df['Month'].tolist()
        self.details["rm_tmp"], self.details["rm_moist"], self.details["rm_pc"], self.details["rm"], self.details["swc"] = [], [], [], [], []
        n_data_month = self.df['Temp'].shape[0]
        self.rate_m_list = [0]*n_data_month
        self.abiotic_factors()
        self.error_journal += f"Dpm_rpm ratio set to {self.dpm_rpm} \n"


        #except:
        #    self.error_journal += 'Error: Year, Month, Temp, Rain, Etp, C_Inp, FYM_Inp, PC or DPM_RPM dont match the appropriate syntax.\n'
        #    return False

        self.rpm = (0.1847 * self.soc + 0.1555) * (self.clay + 1.2750)**( -0.1158)
        self.hum = (0.7148 * self.soc + 0.5069) * (self.clay + 0.3421)**0.0184
        self.bio = (0.0140 * self.soc + 0.0075) * (self.clay+ 8.8473)**0.0567
        self.iom = 0.049 * self.soc**1.139
        self.dpm = self.soc - (self.rpm + self.hum + self.bio + self.iom)

        if self.init_mode == 'weihermuller':
            self.soil_init_carbon = [self.dpm, self.rpm, self.hum, self.bio, self.iom]
           
        elif self.init_mode == 'spin':
            #that modifies self.hum, bio etc to equilibrium
            self.run_equilibrium()

        elif self.init_mode == 'set':
            self.dpm, self.rpm, self.hum, self.bio, self.iom = self.init_set_soc
            self.soc =  self.dpm + self.rpm + self.hum + self.bio + self.iom 
        else: 
            self.error_journal += f'Error: {self.init_mode} init mode does not exit. Choose between weihermuller, spin or set\n'
            return False
        
        return True
       
       
   
    def rmf_tmp(self):
        """Calculates the rate modifying factor for temperature (RMF_Tmp)"""
        if self.temp < -5.0:
            rm_tmp = 0.0
        else:
            rm_tmp = 47.91 / (np.exp(106.06 / (self.temp + 18.27)) + 1.0)
        return rm_tmp

    def rmf_moist(self):
        """Calculates the rate modifying factor for moisture (RMF_Moist)"""
        rmf_max = 1.0
        rmf_min = self.rmf_min
        pE=0.75

        # Calculate soil water functions properties
        M = self.rain - pE * self.pevap  # pE = 0.75 ici     #à comprendre
        bare_factor = 1 if self.pc == 1 else  1/1.8 
        
        """
        min_swc_df = min(0, self.swc + M)
        min_smd_bare_swc = min(self.tsmd_max/1.8, self.swc)
        self.swc = max(self.tsmd_max*bare_factor, min_swc_df)
        """

        if M<0:
            self.swc = max(self.swc + M, self.tsmd_max * bare_factor)
        else:
            self.swc = min(self.swc + M, 0)
        
        if self.swc > 0.444 * self.tsmd_max:
            rm_moist = 1.0
        else:
            rm_moist = (rmf_min + (rmf_max - rmf_min) * (self.tsmd_max - self.swc) / (self.tsmd_max - 0.444 * self.tsmd_max))
        return rm_moist


    


    def rmf_moist_van_Genuchten(self):
        
        """
            Compute Soil moisture factor with Van Genuchten
            a) The soil is allowed to become drier than in the standard version
                of RothC, up to the point which corresponds to a water tension
                of -1000 bar (Mc, capillary water). Between field capacity and permanent wilting point (M), the calculation of the rate modifying factor for moisture is identical to that of the original version of the
                model. Between M and Mc the rate modifying factor remains at
                bmin (currently 0.2), but because the soil is allowed to dry further,
                the amount of water needed to re-wet the soil to the field capacity
                is higher than in the original model, so the rate modifying factor
                can remain at bmin for longer.
                
            b) The hydrological constants (field capacity (-0.05 bar), Mb (-1 bar),
                wilting point M (-15 bar) and Mc (capillary water retained at
                1000 bar)) are calculated using pedotransfer fu
        """
        
        Mb, self.tsmd_max, Mc = calc_Mis(
            silt=self.silt,
            clay=self.clay,
            bulk_density=self.bulk_density,
            oc=self.oc,
            t=1.0,
            theta_R=0.01,
            soil_thickness=self.depth,
        )
    
        """Calculates the rate modifying factor for moisture (RMF_Moist)"""
        rmf_max = 1.0
        rmf_min = 0.1
        pE=0.75

        # Calculate soil water functions properties
        M = self.rain - pE * self.pevap  # pE = 0.75 ici       

        if M<0:
            self.swc = max(self.swc + M, Mc)  # on autorise à s'aecher jusqua Mc
        else:
            self.swc = min(self.swc + M, 0)
        
        if self.swc > Mb:
            rm_moist = 1.0
        elif self.swc > self.tsmd_max:
            rm_moist = (rmf_min + (rmf_max - rmf_min) * (self.tsmd_max - self.swc) / (self.tsmd_max - Mb))
        else:
            rm_moist = rmf_min
        return rm_moist
    
        
    def rmf_pc(self):
        """Calculates the plant retainment modifying factor (RMF_PC)"""
        if self.pc == 0:
            rm_pc = 1.0
        else:
            rm_pc = 0.6
        return rm_pc

        

    def abiotic_factors(self):
        """ Calculate the a*b*c (=self.rate_m) factors for each climatic month"""

        n=self.df['Temp'].shape[0]
        for month in range(n):
            if month % self.time_fact == 0 : self.swc = 0  #hypothese de saturation en janvier
            self.temp = self.df['Temp'][month]
            self.rain = self.df['Rain'][month]
            self.pevap = self.df['ETP'][month]
            self.pc = self.df['PC'][month] 

            rm_tmp = self.rmf_tmp()
            if self.mode == 'RothC_10_N':
                rm_moist = self.rmf_moist_van_Genuchten()
            else: 
                rm_moist = self.rmf_moist()
            rm_pc = self.rmf_pc()
            
            self.details["rm_tmp"].append(rm_tmp)
            self.details["rm_moist"].append(rm_moist)
            self.details["rm_pc"].append(rm_pc)
            self.details["rm"].append(rm_tmp*rm_moist*rm_pc)
            self.details["swc"].append(self.swc)

            self.rate_m_list[month]= rm_tmp*rm_moist*rm_pc


    def decomp(self):
        """Calculates the decomposition"""
        zero = 0e-8
        tstep = 1.0 / self.time_fact
                
        # Decomposition
        dpm1 = self.dpm * np.exp(-self.rate_m * self.dpm_k * tstep)
        rpm1 = self.rpm * np.exp(-self.rate_m * self.rpm_k * tstep)
        bio1 = self.bio * np.exp(-self.rate_m * self.bio_k * tstep)
        hum1 = self.hum * np.exp(-self.rate_m * self.hum_k * tstep)
        
        dpm_d = self.dpm - dpm1
        rpm_d = self.rpm - rpm1
        bio_d = self.bio - bio1
        hum_d = self.hum - hum1
        
        x = 1.67 * (1.85 + 1.60 * np.exp(-0.0786 * self.clay))
        
        # Proportion C from each pool into CO2, BIO and HUM
        dpm_co2 = dpm_d * (x / (x + 1))
        dpm_bio = dpm_d * (self.decomp_ratio_bio / (x + 1))
        dpm_hum = dpm_d * (self.decomp_ratio_hum / (x + 1))
        
        rpm_co2 = rpm_d * (x / (x + 1))
        rpm_bio = rpm_d * (self.decomp_ratio_bio / (x + 1))
        rpm_hum = rpm_d * (self.decomp_ratio_hum / (x + 1))
        
        bio_co2 = bio_d * (x / (x + 1))
        bio_bio = bio_d * (self.decomp_ratio_bio / (x + 1))
        bio_hum = bio_d * (self.decomp_ratio_hum / (x + 1))
        
        hum_co2 = hum_d * (x / (x + 1))
        hum_bio = hum_d * (self.decomp_ratio_bio / (x + 1))
        hum_hum = hum_d * (self.decomp_ratio_hum / (x + 1))
        
        # Update C pools
        self.dpm = dpm1
        self.rpm = rpm1
        self.bio = bio1 + dpm_bio + rpm_bio + bio_bio + hum_bio
        self.hum = hum1 + dpm_hum + rpm_hum + bio_hum + hum_hum
        
        # Split plant C to DPM and RPM
        pi_c_dpm = self.dpm_rpm / (self.dpm_rpm + 1.0) * self.c_inp
        pi_c_rpm = 1.0 / (self.dpm_rpm + 1.0) * self.c_inp
        
        # Split FYM C to DPM, RPM and HUM
        fym_c_dpm = self.fym_dpm_ratio * self.fym
        fym_c_rpm = self.fym_rpm_ratio * self.fym
        fym_c_hum = self.fym_hum_ratio * self.fym
        
        # Add Plant C and FYM_C to DPM, RPM and HUM
        self.dpm = self.dpm + pi_c_dpm + fym_c_dpm
        self.rpm = self.rpm + pi_c_rpm + fym_c_rpm
        self.hum = self.hum + fym_c_hum

        self.co2_emission = dpm_co2 + rpm_co2 + hum_co2 + bio_co2
        
    def gradient_descent(self, step, toc_0, dpm_0, rpm_0, bio_0, hum_0):
        toc_1=0
        tolerance = 0.01  # tC/ha
        max_iterations = 100
        iteration = 0
        carbon_inputs = 0.1
        year_equilibrium = 500
        
        while abs(toc_1 - toc_0) > tolerance and iteration < max_iterations and carbon_inputs < 100:

            # Init
            self.soc,  self.dpm, self.rpm, self.bio, self.hum = toc_0, dpm_0, rpm_0, bio_0, hum_0 
            self.soc=toc_0
            self.c_inp = carbon_inputs /self.time_fact    
            self.fym=0  

            #Run 
            for year in range(year_equilibrium):
                for month in range(self.time_fact): #historic data means
                    id = month
                    self.rate_m = self.rate_m_list[id]
                    self.pc = self.df['PC'][id]
                    self.dpm_rpm = self.df['DPM_RPM'][id]
                    self.decomp()

            #Check if measured SOC is reached
            toc_1 =  self.dpm + self.rpm + self.bio + self.hum + self.iom
            #print(f"Iteration {iteration}: Modeled SOC = {toc_1}, Carbon Inputs = {carbon_inputs}")
            carbon_inputs = carbon_inputs - step * (toc_1-toc_0)
            iteration += 1

        if abs(toc_1 - toc_0) > tolerance or iteration > max_iterations or carbon_inputs > 100:
            return True
        else:
            self.error_journal += f"Equilibrium has reached equilibrium in {iteration} iterations. Carbon input was fixed to {carbon_inputs}. (Gradient Descent step = {step} and tolerance = {tolerance})\n"
            return False


    def run_equilibrium(self):
        """
        Find the pool distribution that reaches the equilibrium with the constraint of a total soil organic content toc = self.soc fixed
        A gradient descent on the input quantity of organic carbon is computed until an equilibrium is reached (more or less the tolerance)
        """

        toc_0, dpm_0, rpm_0, bio_0, hum_0 = self.soc, self.dpm, self.rpm, self.bio, self.hum 
        step = 0.1 
        iteration=0
        while self.gradient_descent(step, toc_0, dpm_0, rpm_0, bio_0, hum_0) and iteration < 100:
            step/=2
            iteration+=1

    
    def run(self):
        """
        Function that calls the self.decomp function for each month of the year during self.n_year
        """        
        year_list = [[0, self.dpm, self.rpm ,self.bio,self.hum, self.iom, self.soc,0, 0]]
        month_list = [[self.df['Year'][0],0, self.dpm, self.rpm ,self.bio,self.hum, self.iom, self.soc, 0, 0]]
        n_data_month = self.df['Temp'].shape[0]
        self.cumul_co2=0
        
        for year in range(self.n_year):
            year_cumul_co2 = 0
            for month in range(self.time_fact):
                id = (year*(self.time_fact) + month) % n_data_month
                self.rate_m = self.rate_m_list[id]
                self.c_inp = self.df['C_inp'][id]
                self.fym = self.df['FYM'][id]
                self.decomp()
                self.soc = self.dpm + self.rpm + self.bio + self.hum + self.iom
                self.cumul_co2 += self.co2_emission
                year_cumul_co2 += self.co2_emission
                month_list.insert(month + self.time_fact*year  +1, [year +self.df['Year'][0],month+1, self.dpm, self.rpm ,self.bio,self.hum, self.iom, self.soc, self.co2_emission,  self.cumul_co2])
                    
            year_list.insert(year +1, [year + self.df['Year'][0],self.dpm, self.rpm ,self.bio,self.hum, self.iom, self.soc, year_cumul_co2, self.cumul_co2])
        
        if not self.dev_mode:
            self.plot_and_save(month_list, year_list)
        else :
            return [row[-1] for row in month_list]


    def plot_and_save(self, month_list, year_list):
        """Plot the result and save it to output_folder"""

        #plot CO2 emission graphs
        emission_month = [row[-2] for row in month_list]
        emission_year = [row[-2] for row in year_list]
        file_path_emission_month = self.output_folder + 'emisiones_C_mensuales'        
        file_path_emission_year = self.output_folder + 'emisiones_C_anuales'
        self.plot_Co2_emissions(emission_month, file_path_emission_month)
        self.plot_Co2_emissions(emission_year, file_path_emission_year)
        self.error_journal += f"Mean soc_year emissions: {sum(emission_year)/len(emission_year)}\n"

        #save into excel : 
        output_years = pd.DataFrame(year_list, columns=["Year","DPM_t_C_ha","RPM_t_C_ha","BIO_t_C_ha","HUM_t_C_ha","IOM_t_C_ha", "SOC_t_C_ha", "C_t_emissions", "Cumul_C_t_emissions"])     
        output_months = pd.DataFrame(month_list, columns=["Year","Month","DPM_t_C_ha","RPM_t_C_ha","BIO_t_C_ha","HUM_t_C_ha","IOM_t_C_ha", "SOC_t_C_ha", "C_t_emissions", "Cumul_C_t_emissions" ])
        
        output_years.to_excel(self.output_yearly_name , index = False)
        output_months.to_excel(self.output_monthly_name,  index = False)
        df_details = pd.DataFrame(self.details)
        df_details.to_excel(self.output_folder + "details.xlsx", index=False)

        # Plot Carbon stock evolution per pool
        fig, ax = plt.subplots(1, 1, figsize=(30, 14))
        colors = ['yellow', 'orange', 'green', 'brown', 'gray']
        output_months[["DPM_t_C_ha","RPM_t_C_ha","BIO_t_C_ha","HUM_t_C_ha", "IOM_t_C_ha"]].plot(kind='bar', stacked=True, color=colors, ax=ax)
        ax.set_ylabel('C stocks (Mg/ha)')
        ax.set_xlabel('Month')
        xticks = ax.get_xticks()
        ax.set_xticks(xticks[::self.time_fact])
        ax.set_xticklabels(output_months.index[::self.time_fact]//self.time_fact + self.df['Year'][0], rotation=45)

        ax.set_title('Evolución en las formas de materia orgánica del suelo (ton ha⁻¹)')
        ax.legend(['DPM', 'RPM', 'BIO', 'HUM', 'IOM'], loc='upper left', bbox_to_anchor=(1,1))
        plt.savefig(self.output_folder + 'carbon_stocks_per_pool.png')

        with open(self.output_folder + "journal.txt", 'w') as file:
            file.write(self.error_journal)
        print(self.error_journal)



    def plot_Co2_emissions(self, emission, file_path):
 
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(emission)), emission, linestyle='-', color='b', label='C Emission')
        plt.xlabel('time (Year or Month)')
        plt.ylabel('C Emission')
        plt.title('C Emission Over time')
        plt.legend()
        if not self.dev_mode : plt.savefig(file_path)

