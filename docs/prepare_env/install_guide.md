conda create -n geneface python=3.9 -y
conda activate geneface
# for newer GPUs (e.g., RTX3090)
conda install pytorch=1.12 torchvision cudatoolkit=11.3 -c pytorch -c nvidia -y
# for older GPUs (e.g., RTX2080)
# install pytorch-3d
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
conda install pytorch3d=0.7.2 -c pytorch3d -y
# other dependency
pip install -r docs/prepare_env/requirements.txt 