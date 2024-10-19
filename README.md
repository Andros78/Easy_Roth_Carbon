﻿# RothC_version_andres
 
 Python version of The Rothamsted carbon model (RothC) 26.3.
 RothC is a model for the turnover of organic carbon in non-waterlogged topsoil that allows for the effects of soil type, temperature, soil moisture and plant cover on the turnover process.

**1. Input excel file**
All you have to do is put your excel file in input_data folder. The input excel file has the following columns:
    Year - 
    Month - 
    Temp - (month average) 
    Rain - (mm)
    ETP - monthly 
    Cinp - the carbon input monthly (tC/ha)
    FYM - carbon from farm manure (tC/ha)
    PC - Plant cover (a boolean 1 if there is a plant cover and 0 if not)
    DPM_RPM - ratio dpm/rpm that is the mean of the weighted ratios of all the carbon input types. (see Coleman)
    Clay - percentage of clay (between 0 and 100)
    Depth - cm of depth of soil
    SOC - Soil Organic Carbon at the initialisation of the prediction (the year 0) in tc/ha (over the first 'Depth' cm)

**2. Check the config** 
In the config file, you have to indicate which excel file and which sheet in the excel file have to be selected by the model. Other parameters can be modified:
    'nbr_year' : '30', : number of year of the simulation
    'init_mode' : 'spin', init mode is to choose between Weihermuller, Spin and Set. 
    
Weihermuller mode (see Weihermuller et al. 2013):         
self.rpm = (0.1847 * self.soc + 0.1555) * (self.clay + 1.2750)**( -0.1158)
self.hum = (0.7148 * self.soc + 0.5069) * (self.clay + 0.3421)**0.0184
self.bio = (0.0140 * self.soc + 0.0075) * (self.clay+ 8.8473)**0.0567
self.iom = 0.049 * self.soc**1.139
self.dpm = self.soc - (self.rpm + self.hum + self.bio + self.iom)

Spin mode finds a equilibrium in the soil dynamics. It is a descent gradient on the carbon inputs. At the n-th iteration, it runs the model for 500 years and if it reaches the initial SOC with an error less than 0.01, then the pool (rpm, hum, bio iom, dpm) are the ones use to init the model. If it does not reach SOC,   carbon_inputs (n+1) = carbon_inputs(n) - step * (toc_1-toc_0).

Set mode is a mode in the which you set all the pools manually in the 'init_set_soc' : \[dpm, rpm, hum, bio, iom, \]

    'swc' : '0', soil water content in january
    'SHOW':0, if you want to show the graphics set it to 1. if not set it to 0

    Hyper-parameters: 
  --> rmf_min : default set to 0.2 
  --> decomp_params : default set to [10, 0.3, 0.66, 0.02]
  --> decomp_ratio : [0.46, 0.54]
  --> fym_ratio : [0.49, 0.49, 0.02]

  **3.Run main.py**
  Just run it :)


  
  More details on Parameters: 
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



      
