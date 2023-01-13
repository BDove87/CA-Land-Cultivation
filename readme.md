# California Cultivated Land Prediction

This project uses eo-learn and Scikit-Learn to predict whether or not an area is cultivated or not via Sentinelhub imagery. The code generates eo-patches which are too large to upload to the github repository. The repository contains the final trained model as well as a number of graphs and geotiffs, the final prediction graph being the most useful. The geotiffs look black but information can be extracted from them.

This project is based off a tutorial from eo-learn (https://eo-learn.readthedocs.io/en/latest/examples/land-cover-map/SI_LULC_pipeline.html)

## Requirements

eo-learn

Reference data for training - this project was trained using data from the California Natural Resources Agency (https://data.cnra.ca.gov/dataset/statewide-crop-mapping)