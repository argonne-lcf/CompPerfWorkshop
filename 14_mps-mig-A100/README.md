# Nvidia Multi-Instance GPU (MIG) mode:

MIG mode will enable multi-tenancy on a GPU where different instances can run simultaneously on a GPU by sharing the resources. This capability will improve
the utilization of the GPUs by running either separate applications or multiple instances of single application.

Currently, a GPU can be partitioned in 5 ways, with varying amount of resources in each. A GPU instance (GI) denotes the partitioned GPU resource and Compute
instance (CI) denotes the compute resources in each GI. 

MIG user guide is available at https://docs.nvidia.com/datacenter/tesla/mig-user-guide/index.html

## Running applications with MIG-mode on ThetaGPU:


**Step (1):**  
Log onto a compute node with mig-mode attribute set to True

`qsub -I -t 60 -n1 -A project -q queue --attrs mig-mode=True`


check if MIG mode is enabled with 'nvidia-smi', this should list the GPUs followed by a table of MIG devices. To see the list of all possible GPU instances, use 

`nvidia-smi mig -lgip` 

For a specific GPU, pass its id with "-i" option; for GPU 0, use 


```
memani@thetagpu05:~$ nvidia-smi mig -i 0 -lgip
+--------------------------------------------------------------------------+  
| GPU instance profiles:                                                   |  
| GPU   Name          ID    Instances   Memory     P2P    SM    DEC   ENC  |  
|                           Free/Total   GiB              CE    JPEG  OFA  |  
|==========================================================================|  
|   0  MIG 1g.5gb     19     0/7        4.75       No     14     0     0   |  
|                                                          1     0     0   |  
+--------------------------------------------------------------------------+  
|   0  MIG 2g.10gb    14     0/3        9.75       No     28     1     0   |  
|                                                          2     0     0   |  
+--------------------------------------------------------------------------+  
|   0  MIG 3g.20gb     9     0/2        19.62      No     42     2     0   |  
|                                                          3     0     0   |  
+--------------------------------------------------------------------------+  
|   0  MIG 4g.20gb     5     0/1        19.62      No     56     2     0   |  
|                                                          4     0     0   |  
+--------------------------------------------------------------------------+  
|   0  MIG 7g.40gb     0     0/1        39.50      No     98     5     0   |  
|                                                          7     1     1   |  
+--------------------------------------------------------------------------+
```

This shows that GPU 0 can be split into MIG devices in 5 ways with different number of instances in each (column 4). To see the placement choices, you can use ‘lgipp’ option


```
memani@thetagpu05:~$ nvidia-smi mig -i 0 -lgipp  
GPU  0 Profile ID 19 Placements: {0,1,2,3,4,5,6}:1  
GPU  0 Profile ID 14 Placements: {0,2,4}:2  
GPU  0 Profile ID  9 Placements: {0,4}:4  
GPU  0 Profile ID  5 Placement : {0}:4  
GPU  0 Profile ID  0 Placement : {0}:8  
```


**Step (2)**   
Create GPU Instances (GI) with ‘cgi’ option, for example, to split GPU 0 into 2 instances with 20 GB memory for each, use

`nvidia-smi mig -i 0 -cgi 9,9`

where the parameter to create GPU instance ‘cgi’ is the gpu instance profile id in column 3 above. The name of the instance like ‘MIG 3g.20gb’ can also be used instead. The parameter ‘lgi’ can be used to list the GPU instances


```
memani@thetagpu05:~$ nvidia-smi mig -i 0 -cgi 9,9
Successfully created GPU instance on GPU  0 using profile ID  9
Successfully created GPU instance on GPU  0 using profile ID  9

memani@thetagpu05:~$ nvidia-smi mig -i 0 -lgi
+----------------------------------------------------+
| GPU instances:                                     |
| GPU   Name          Profile  Instance   Placement  |
|                       ID       ID       Start:Size |
|====================================================|
|   0  MIG 3g.20gb       9        1          4:4     |
+----------------------------------------------------+
|   0  MIG 3g.20gb       9        2          0:4     |
+----------------------------------------------------+
```


Next create Compute Instance (CI) with ‘cci’ parameter for each GPU instance

```
memani@thetagpu05:~$ nvidia-smi mig -i 0 -gi 1 -cci 0
Successfully created compute instance on GPU  0 GPU instance ID  1 using profile ID  0

memani@thetagpu05:~$ nvidia-smi mig -i 0 -gi 2 -cci 0
Successfully created compute instance on GPU  0 GPU instance ID  2 using profile ID  0
```


Verify if the compute instances are created with ‘lci’ option

```
memani@thetagpu05:~$ nvidia-smi mig -i 0 -lci
+-------------------------------------------------------+
| Compute instances:                                    |
| GPU     GPU       Name             Profile   Instance |
|       Instance                       ID        ID     |
|         ID                                            |
|=======================================================|
|   0      1       MIG 1c.3g.20gb       0         0     |
+-------------------------------------------------------+
|   0      2       MIG 1c.3g.20gb       0         0     |
+-------------------------------------------------------+
```


**Step (3)**  
To run the codes in MIG mode, get the Unique ID for each MIG device with “-L” option to nvidia-smi. The new format follows this convention: `MIG-<GPU-UUID>/<GPU instance ID>/<compute instance ID>`.

```
memani@thetagpu05:~$ nvidia-smi -L
GPU 0: A100-SXM4-40GB (UUID: GPU-2fa5cf01-9520-1bc1-995b-cc0c6d9bca7c)
  MIG 1c.3g.20gb Device 0: (UUID: MIG-GPU-2fa5cf01-9520-1bc1-995b-cc0c6d9bca7c/1/0)
  MIG 1c.3g.20gb Device 1: (UUID: MIG-GPU-2fa5cf01-9520-1bc1-995b-cc0c6d9bca7c/2/0)
```

Use the UUID to CUDA_VISIBLE_DEVICES environment variable as
`CUDA_VISIBLE_DEVICES=UUID <application> &`

For example, to run two apps myapp1.py and myapp2.py

```
CUDA_VISIBLE_DEVICES=MIG-GPU-2fa5cf01-9520-1bc1-995b-cc0c6d9bca7c/1/0 python myapp1.py &
CUDA_VISIBLE_DEVICES=MIG-GPU-2fa5cf01-9520-1bc1-995b-cc0c6d9bca7c/2/0 python myapp2.py &
```

Verify if both are running on two instances with ‘nvidia-smi’ under the Processes tab

** Step (5)**  
To destroy the instances, first start with compute instances (CI) then GPU instances (GI)

```
memani@thetagpu05:~$ nvidia-smi mig -i 0 -gi 1 -dci
Successfully destroyed compute instance ID  0 from GPU  0 GPU instance ID  1

memani@thetagpu05:~$ nvidia-smi mig -i 0 -gi 2 -dci
Successfully destroyed compute instance ID  0 from GPU  0 GPU instance ID  2

memani@thetagpu05:~$ nvidia-smi mig -i 0 -dgi
Successfully destroyed GPU instance ID  1 from GPU  0
Successfully destroyed GPU instance ID  2 from GPU  0
```


Verify if the MIG devices are torn down with nvidia-smi
```
+-----------------------------------------------------------------------------+
| MIG devices:                                                                |
+------------------+----------------------+-----------+-----------------------+
| GPU  GI  CI  MIG |         Memory-Usage |        Vol|         Shared        |
|      ID  ID  Dev |                      | SM     Unc| CE  ENC  DEC  OFA  JPG|
|                  |                      |        ECC|                       |
|==================+======================+===========+=======================|
|  No MIG devices found                                                       |
+-----------------------------------------------------------------------------+
```


