# Create a virtual environment
/lus/theta-fs0/software/datascience/conda/2021-09-22/mconda3/bin/python -m venv env
source env/bin/activate
python -m pip install --upgrade pip

# Install Balsam
python -m pip install balsam

# Login to the Balsam server. This will prompt you to visit an ALCF login page; this command will
# block until you've logged in on the webpage.
balsam login

# Load the cobalt-gpu module, so Balsam submits to the ThetaGPU queues
module load cobalt/cobalt-gpu

# Create a Balsam site
# Note: The "-n" option specifies the site name; the last argument specifies the directory name
balsam site init -n thetagpu_tutorial thetagpu_tutorial
cd thetagpu_tutorial
balsam site start
cd ..
