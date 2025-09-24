mamba install -c conda-forge gcc=10.4.0
mamba install -c conda-forge gxx_linux-64=10.4.0
mamba install -c nvidia/label/cuda-11.7.1 cuda
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install "fair-esm[esmfold]" torch==1.12.1+cu113
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307' --no-cache-dir
pip install pytorch_lightning==1.5
pip install "alphafold @ git+https://github.com/google-deepmind/alphafold.git@v2.3.1" "dm-haiku @ git+https://github.com/google-deepmind/dm-haiku@v0.0.13" "jax[cuda12]==0.4.30"
pip install pydssp ProDy numpy==1.26.0
pip install -e .
