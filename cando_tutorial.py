"""
CANDO Tutorial: Drug Repurposing Analysis
==========================================
This script demonstrates basic usage of the CANDO platform for drug repurposing analysis.
CANDO (Computational Analysis of Novel Drug Opportunities) uses drug-protein interaction
signatures to identify potential therapeutic applications for existing drugs.

Please refer the jupyter notebook for a more detailed tutorial. https://github.com/ram-compbio/CANDO/blob/master/CANDO_tutorial.ipynb
"""

import cando as cnd  # Import CANDO library with shortened alias for convenience
import os
import json

# =============================================================================
# Environment Setup and Data Preparation
# =============================================================================

# Ensure we're in the correct directory structure
# If script is run multiple times, navigate up from any nested tutorial folders
while os.getcwd().split(os.sep)[-1] == 'tutorial': 
    os.chdir('..')

# Download and prepare tutorial data
cnd.get_tutorial()  # Download tutorial files and example datasets
cnd.get_data(v='test.0', org='tutorial')  # Get specific test data version
os.chdir("tutorial")  # Navigate into the tutorial working directory

# =============================================================================
# Data File Configuration
# =============================================================================

# Define file paths for the core CANDO data files
cmpd_map = 'cmpds-v2.2.tsv'           # Compound mapping file: drug identifiers and metadata
ind_map = 'cmpds2inds-v2.2.tsv'       # Association file: drug-indication relationships
matrix_file = 'tutorial_matrix-approved.tsv'  # Interaction matrix: drug-protein binding signatures

# =============================================================================
# Analysis Parameters
# =============================================================================

# Configuration parameters for similarity analysis
dist_metric = 'cosine'  # Distance metric for comparing drug interaction signatures
ncpus = 1  # Number of CPU cores to use for parallel processing

# =============================================================================
# CANDO Object Initialization and Analysis
# =============================================================================

# Create main CANDO object with all data and compute drug-drug similarities
cando = cnd.CANDO(cmpd_map, ind_map, matrix=matrix_file, compound_set='approved', 
                  compute_distance=True, dist_metric=dist_metric, ncpus=ncpus)

# =============================================================================
# Data Exploration and Results
# =============================================================================

# Display basic statistics about the loaded dataset
print(len(cando.compounds), 'compounds')     # Total number of drugs in the dataset
print(len(cando.indications), 'indications') # Total number of medical conditions
print(len(cando.proteins), 'proteins')       # Total number of protein targets

# =============================================================================
# HIV Case Study: Finding Associated Compounds (Similarly for other diseases)
# =============================================================================

# Search for HIV-related indications in the database
print(cando.search_indication('HIV'))

# Retrieve the specific HIV indication object using its MESH identifier
hiv = cando.get_indication("MESH:D015658")
# Display the number of compounds currently associated with HIV treatment
print(len(hiv.compounds), 'compounds are associated with', hiv.name)

# Predict the top 10 indications for the hiv drug
cando.canpredict_compounds("MESH:D015658", n=10, topX=10)
