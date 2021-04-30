# Hands on for Data Parallel Deep Learning on ThetaGPU

1. Request an interactive session on ThetaGPU:

   ```bash
   # Login to theta
   ssh -CY user@theta.alcf.anl.gov
   # Login to thetaGPU login node
   ssh -CY thetagpusn1
   # Request an interactive job with 1 node
   qsub -n 1 -q training -a comp_perf_workshop -I -t 2:00:00
   ```

2. Copy the 