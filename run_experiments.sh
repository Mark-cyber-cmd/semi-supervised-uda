echo "Activating conda env..."
. /mnt/data/optimal/dingyiming/miniconda3/etc/profile.d/conda.sh
# 验证自己的conda是否激活
which conda
# 激活需要的环境
conda activate DAFormer

# echo "Prepared Datasets"
# python tools/convert_datasets/dsec.py data/DSEC_Semantic --nproc 8

echo "Begin training!"
cd /mnt/data/optimal/dingyiming/Codes/semi-supervised-uda
# cityscapes ------->  DSEC
# python run_experiments.py --config configs/xxformer/cs2dsec_uda_xxformer.py
# GTA5       ------->  DSEC
python run_experiments.py --config configs/xxformer/gta2dsec_semi_xxformer.py
