DATA_FILENAME='HIGGS.csv'
DATA_DIR='/data/higgs_files'
LAB_COL=0
NUM_TREES=100
NUM_TEST=1000

#python3 train_polytope_new.py --labelcol $LAB_COL --datadir $DATA_DIR --datafilename $DATA_FILENAME --numtrees $NUM_TREES --numtest $NUM_TEST
TEST_FNAME="$DATA_DIR"
TEST_FNAME+="test_"
TEST_FNAME+="$DATA_FILENAME"

#POLY_FNAME="$DATA_DIR"
#POLY_FNAME+="poly_"
#DATA_STR="${DATA_FILENAME%.*}"
#POLY_FNAME+="$DATA_STR"
#POLY_FNAME+=".json"





POLY_FNAME='/data/poly_SUSY.csv'
DIST='ext'
MEAS='topk'
TOPK=2000
TOPM=11
./exe2 --cardfilename '/data/higgs_files/card_list.csv' --poolfilename '/data/higgs_files/pooled.csv' --classfilename '/data/higgs_files/class_list.csv' --meanfilename '/data/higgs_files/mean_list.csv' --hyperfilename '/data/higgs_files/hyperrectangles_HIGGS_maxdepth2.csv' --featurefilename '/data/higgs_files/features_4.csv' --writefname '/deta/hilbert_test_susy2.csv' --writefname2 '/data/hilbert_test_susy3.csv' --newfilename '/data9/newpolytopes.csv' --oversample "/data9/oversample.csv" --undersample "/data9/undersample.csv" --polymodelfilename $POLY_FNAME --numthreads 1 --datafilename '/data/higgs_files/test_HIGGS.csv' --distance $DIST --measure $MEAS --topm $TOPM --topk $TOPK
