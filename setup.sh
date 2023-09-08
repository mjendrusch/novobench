conda create -n novobench python=3.9
conda activate novobench
conda install -c conda-forge aria2
module load CUDA/11.3.1
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install "fair-esm[esmfold]"
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'
pip install pytorch_lightning==1.5.10
pip install pydssp
pip install ProDy
git clone https://github.com/facebookresearch/esm.git
cd esm
pip install -e .
cd ..
git clone https://github.com/deepmind/alphafold.git
cd alphafold
pip install -e .
bash scripts/download_alphafold_params.sh .
cd ..
pip install -e .
