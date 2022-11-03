DATA_FILENAME='sgemm.csv'
DATA_DIR='/data2/sgemm/'

TEST_FNAME="$DATA_DIR"
TEST_FNAME+="test_sgemm_t.csv"
#TEST_FNAME+="$DATA_FILENAME"

CARD_FNAME="$DATA_DIR"
CARD_FNAME+='card_list.csv'

CLASS_FNAME="$DATA_DIR"
CLASS_FNAME+='class_list.csv'

MEAN_FNAME="$DATA_DIR"
MEAN_FNAME+='mean_list_t.csv'

HYPER_FNAME="$DATA_DIR"
HYPER_FNAME+='hyper_t.csv'

POOL_FNAME="$DATA_DIR"
POOL_FNAME+='pooled.csv'

FEATURE_FNAME="$DATA_DIR"
FEATURE_FNAME+='features.csv'

./exe2 --writedirname $DATA_DIR --datafilename $TEST_FNAME --cardfilename $CARD_FNAME --classfilename $CLASS_FNAME --meanfilename $MEAN_FNAME --hyperfilename $HYPER_FNAME --poolfilename $POOL_FNAME  --featurefilename $FEATURE_FNAME 
