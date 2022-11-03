# T-Rex

All the datasets can be downloaded from https://archive.ics.uci.edu/ml/index.php .

1) model_training/ : Train the model in sckit-learn and output the hyperrectangles and centroid. Needs a custom fork of scikit-learn (zip file attached). train_models_run.sh is an example script to run this
2) index_creation/ : Generate hilbert index. cmake . & make. full_runscript_<data>.sh contain examples to run
3) inference/ : Perform Inference. 
