# Create a virtual environment
/lus/theta-fs0/software/datascience/conda/2021-09-22/mconda3/bin/python -m venv env
source env/bin/activate
python -m pip install --upgrade pip

# Install Balsam
python -m pip install balsam

# Create a Balsam site
balsam site init -n thetagpu_tutorial thetagpu_tutorial
pushd thetagpu_tutorial
balsam site start
popd
