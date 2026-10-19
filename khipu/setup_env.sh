#!/bin/bash
# Creates the conda environment "tp53dl" on the Khipu cluster (UTEC). Run it ONCE on the login node, the only node with internet:
#   bash ~/tp53-transfer-learning/khipu/setup_env.sh
# Versions match the Mac. scikit-learn 1.7.1 is needed to open elasticnet_fuente.joblib.

# No 'set -u': Lmod on Khipu reads LD_PRELOAD without defining it.
set -eo pipefail
export LD_PRELOAD="${LD_PRELOAD:-}"

module load miniconda/3.0
eval "$(conda shell.bash hook)"

if ! conda env list | grep -q "^tp53dl "; then
    conda create -y -n tp53dl --override-channels -c conda-forge \
        python=3.11 pandas=2.3.2 pyarrow=21.0.0 scikit-learn=1.7.1 joblib matplotlib nbconvert nbformat ipykernel
fi

conda activate tp53dl
# Networks (notebooks 06–08): Keras on PyTorch, same versions as the Mac, CPU only.
pip install --quiet --index-url https://download.pytorch.org/whl/cpu "torch==2.14.1+cpu"
pip install --quiet keras==3.15.1
python -c "import sklearn, pandas, pyarrow, matplotlib, nbconvert; print('OK: scikit-learn', sklearn.__version__, '· pandas', pandas.__version__)"
KERAS_BACKEND=torch python -c "import keras, torch; print('OK: keras', keras.__version__, '· torch', torch.__version__)"
