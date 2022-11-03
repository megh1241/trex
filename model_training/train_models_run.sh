
DATA_FILENAME='foo.csv'
DATA_DIR='/data/'
LAB_COL=100
NUM_TREES=100
NUM_TEST=1000
NUM_FEAT=7
python3 distribution_train_hyper_prune.py --labelcol $LAB_COL --datadir $DATA_DIR --datafilename $DATA_FILENAME --numtrees $NUM_TREES --numtest $NUM_TEST
