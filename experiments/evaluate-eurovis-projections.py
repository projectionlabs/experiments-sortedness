import pandas as pd
import os
import glob

techs = ["GlimmerMDS", "PCA", "RobPCA", "TSNE"]
datasets = ['Entangled1-3d-3cl-separate', 'Entangled1-3d-4cl-separate', 'Entangled1-3d-5cl-separate'
, 'boston', 'breastCancer-diagnostic', 'entangled2-4d-overlap', 'entangled2-5d-adjacent', 'entangled2-5d-overlap'
, 'entangled2-6d-adjacent', 'entangled3-l-3d-smallOverlap', 'entangled3-m-3d-adjacent', 'entangled3-m-3d-smallOverlap'
, 'entangled3-s-3d-adjacent', 'entangled3-s-3d-bigOverlap', 'fisheriesEscapementTarget', 'fisheriesHarvestRule'
, 'gauss-n100-10d-3largeCl', 'gauss-n100-10d-3smallCl', 'gauss-n100-10d-5largeCl', 'gauss-n100-10d-5smallCl'
, 'gauss-n100-5d-3largeCl', 'gauss-n100-5d-3smallCl', 'gauss-n100-5d-5largeCl', 'gauss-n100-5d-5smallCl'
, 'gauss-n500-10d-3smallCl', 'gauss-n500-10d-5largeCl', 'gauss-n500-10d-5smallCl', 'gauss-n500-5d-3largeCl'
, 'gauss-n500-5d-3smallCl', 'gauss-n500-5d-5largeCl', 'gauss-n500-5d-5smallCl', 'industryIndices'
, 'iris', 'musicnetgroups', 'olive', 'tse300', 'wine', 'world-12d', 'worldmap']

# use glob to get all the csv files 
# in the folder
# path = os.getcwd()
path_projs = "/home/tacito/Documents/Datasets/EUROVIS_projs"
csv_files = glob.glob(os.path.join(path_projs, "*.csv"))

# datasets = set()
# proj_techs = set()

files_dict = {}

for f in csv_files:

    file_tokens = f.split("/")
    file_name = file_tokens[-1]
    file_name_tokens = file_name.split("_")

    if (file_name_tokens[0] in datasets) and (file_name_tokens[1] in techs):
        # read the csv file
        df = pd.read_csv(f)
