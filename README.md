# Person-Tracking-and-Re-ID
Person Tracking and Person Re-identification for Surveillance in Multi-Camera Systems

### README in progress. Please visit Webpage for more details

Pre-requisite - Clone this repository at a suitable location

## Steps for running the Tracking program:
1. Make by setting appropriate variables in Makefile inside darknet folder 
2. Chnage working directory to 'people-counting' and empty the 'gallery' folder and 'count.txt' for metadata
3. Run python script 'darknet_people_counter.py' parsing necessary arguments to begin tracking
4. Output will be stored as 'out.mp4' by default
5. To use MobileNetSSD for person detection instead of darknet, use 'people_counter.py' in step 3

## Steps for running the Re-ID program:
1. Change current directory to 'deep-reid'
2. Perform 'pip install requirements.txt' and run 'setup.py' inside deep-reid folder to install and setup torchreid
3. Download weights and store in 'log/resnet50' folder
4. Setup (metadata) images in 'gallery' and 'query' folders inside 'reid-data/Test_folder' according to testing requirements
5. Run script 'finder.py' to perform Person Re-identification
6. Output can be visualized in location: 'log/resnet50/visrank_Test_dataset/'

## Run it on colab:
Colaboratory: https://colab.research.google.com/drive/1P2_OQ1ERhmGr5TNIkzMKqqhH7ss2LaQs?usp=sharing

## For more details:

Webpage: http://homepages.iitb.ac.in/~15d100004/

Github Webpage: https://nikunjmshah.github.io/Person-Tracking-and-Re-ID/
