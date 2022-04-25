NETWORK=$1
N_BIT=$2
QUAN_DOWNSAMPLE=$3
LOG_DIR=log/${NETWORK}_${N_BIT}bit_quantize_downsample_${QUAN_DOWNSAMPLE}
mkdir -p ${LOG_DIR}
python3 train.py --data=/home/zechun/imagenet --batch_size=256 --learning_rate=1.25e-3 --epochs=128 --weight_decay=0 --student=${NETWORK} --n_bit=${N_BIT} --quantize_downsample=${QUAN_DOWNSAMPLE} | tee -a ${LOG_DIR}/training.txt
