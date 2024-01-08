# Leaf_burn
Simulating the canopy temperature of potted grapevines during the heat wave of June 28th 2019. The results are published in a paper entitled "Clusters of grapevine genes in a burning world" in the New Phytologist journal by [Coupel-Ledru et al. (2024)](https://doi.org/10.1111/nph.19540).


# Installation

## Create and set `conda` environment

This will take several minutes to create the ~3.4 GB environment

    conda create -n leaf_burn -c openalea3 -c conda-forge openalea.hydroshoot=5.2.2
    conda activate leaf_burn


## Clone and install the project package

Using for example the HTTPS protocol

    https://github.com/awestgeest/Leaf_burn.git

Go to the project root directory

    cd ~/Leaf_burn
    pip install -e .

## Running the simulations

### 1. Create preprocessed inputs

By executing

    python main_preprocess.py


This will create a directory named `preprocessed_data` having 5 subdirectories called after the digitized plants
('belldenise', 'plantdec', 'poulsard', 'raboso', 'salice').
Within each of these 5 subdirectories will be created the following files:

- `<plant_id>_dynamic.json`: hourly data of incident and absorbed irradiance per leaf and diffuse/global irradiance
ratio
- `<plant_id>_static.json`: form factors and nitrogen content (gN/m2leaf) for each leaf
- `initial_mtg_<plant_id>.pckl`: initialized mtg object 
- `geometry_<plant_id>.bgeom`: geometry objects of the mtg object


### 2. Create preprocessed inputs

By executing

    python main_process.py


This will run hydroshoot on the preprocessed data. The directory `output` will be created with 4 subdirectories named
after the scenarios of stomatal conductance behavior ('biochemical_dominant', 'intermediate',
'stomatal_sensitivity_dominant', 'stomatal_sensitivity_weak').

Within each of the latter 4 directories, 4 other subdirectories will be created following the temperature and
humidity profiles simulated ('tcst_ucst', 'tcst_uvar', 'tvar_ucst', 'tvar_uvar').

Finally, wihtin each of the latter directories, five subdirectories will be created, named after the digitized plants:
('belldenise', 'plantdec', 'poulsard', 'raboso', 'salice'), which each dubsirectory, 24 pckl files will be created
(leaf-level outputs for each of 24 hours simulated) in addition to 'time_series.csv' (canopy-level outputs).


### 3. Create figures

By running

    python main_analysis.py


This will create hourly temperature profile figures for each of stomatal behavior scenarios in 
addition to other graphs representing the Arrhenius reponse function of photosynthesis to leaf temperature
and the absorbed irradiance of the 5 plants on gournd basis (W/m2ground) on hourly time steps.


## You're done!
