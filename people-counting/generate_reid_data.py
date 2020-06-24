

import os
import math
import shutil

dataset_dir = '../deep-reid/reid-data/Test_folder'

# remove and create dataset directory
shutil.rmtree(dataset_dir, ignore_errors=True)
os.mkdir(dataset_dir)
os.mkdir(dataset_dir + '/gallery')
os.mkdir(dataset_dir + '/query')

for f in (os.listdir('gallery')):
    shutil.copy('gallery/' + f, dataset_dir + '/gallery')
