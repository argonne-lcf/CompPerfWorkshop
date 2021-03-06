#!/bin/bash -x 

# Create the Hello app
python hello.py

# List apps
balsam app ls --site thetagpu_tutorial

# Create a Hello job
# Note: tag it with the key-value pair workflow=hello for easy querying later
balsam job create --site thetagpu_tutorial --app Hello --workdir=demo/hello --param say_hello_to=world --tag workflow=hello --yes

# The job resides in the Balsam server now; list the job
balsam job ls --tag workflow=hello
