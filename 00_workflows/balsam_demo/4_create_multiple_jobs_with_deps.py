#!/usr/bin/env python
from balsam.api import Job,BatchJob

# Create a collection of jobs, each one depending on the job before it
n=0
job = Job( site_name='thetagpu_tutorial',
           app_id="Hello", 
           workdir=f"demo/hello_deps{n}", 
           parameters={"say_hello_to": f"world {n}!"},
           tags={"workflow":"hello_deps"}, 
           node_packing_count=8, 
           gpus_per_rank=1)
job.save()
for n in range(7):
    job = Job( site_name='thetagpu_tutorial', 
               app_id="Hello", 
               workdir=f"demo/hello_deps{n}", 
               parameters={"say_hello_to": f"world {n}!"},
               tags={"workflow":"hello_deps"}, 
               node_packing_count=8, 
               gpus_per_rank=1, 
               parent_ids=[job.id])  # Sets a dependency on the prior job
    job.save()

# Create a BatchJob to run jobs with the workflow=hello_deps tag
BatchJob.objects.create(
    site_id=287,
    num_nodes=1,
    wall_time_min=10,
    queue="single-gpu",
    project="datascience",
    job_mode="mpi",
    filter_tags={"workflow":"hello_deps"},
    #queue="full-node",
)

