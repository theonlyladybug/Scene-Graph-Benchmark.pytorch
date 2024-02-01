## Installation

Most of the requirements of this projects are exactly the same as [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). If you have any problem of your environment, you should check their [issues page](https://github.com/facebookresearch/maskrcnn-benchmark/issues) first. Hope you will find the answer.

### Requirements:
- Python <= 3.8 (3.8.0)
- PyTorch >= 1.2 (Mine 1.4.0 (CUDA 10.1)) (1.10.1)
- torchvision >= 0.4 (Mine 0.5.0 (CUDA 10.1)) (0.11.2)
- cuda-toolkit 11.3.1 (both binary and compiler)
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV


### Step-by-step installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name scene_graph_benchmark
conda activate scene_graph_benchmark

# this installs the right pip and dependencies for the fresh python
conda install ipython
conda install scipy
conda install h5py

# scene_graph_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python overrides

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 10.1
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch

export INSTALL_DIR=$PWD

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex

# WARNING if you use older Versions of Pytorch (anything below 1.7), you will need a hard reset,
# as the newer version of apex does require newer pytorch versions. Ignore the hard reset otherwise.
# this line is still necessary for newer pytorch, not sure why
git reset --hard 3fe10b5597ba14a748ebb271a6ab97c09c5701ac 

python setup.py install --cuda_ext --cpp_ext


# install PyTorch Detection
cd $INSTALL_DIR
git clone https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch.git
cd scene-graph-benchmark

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop


unset INSTALL_DIR
```

### Other issues for customized images
1. module import errors

- Packages like numpy will have some submodules not found. Just follow the errors and install these packages.
- After apex building:
```
/mnt/scratch/fast0/yiran/BLIP/sgb/apex/apex/amp/_amp_state.py
```
will include a package, "container_abcs" which will raise some errors.
Current lines:
```
import os
import torch

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

if TORCH_MAJOR == 0:
    import collections.abc as container_abcs
else:
    from torch._six import container_abcs
```
won't raise any errors. Remember to take care of this after each rebuild.
- Line 4 of maskrcnn_benchmark/utils/imports.py: 
```
if torch._six.PY3:
```
changed to
```
if torch._six.PY37:
```
2. problems with paths
- `/mnt/scratch/fast0/yiran/BLIP/sgb/Scene-Graph-Benchmark.pytorch/maskrcnn_benchmark/data/datasets/visual_genome.py`: will raise an assertion error at Line 320-ish. This is because the code is still trying to find the VG images for training and testing but we are not using those data. (We are testing on our own images instead.) Therefore, change these lines into:
```
        # if os.path.exists(filename):
            # fns.append(filename)
            # img_info.append(img)
        fns.append(filename)
        img_info.append(img)
        
    print(len(fns))
    print(len(img_info))
    # assert len(fns) == 108073
    # assert len(img_info) == 108073
    return fns, img_info
```
- Change the root path of datasets in Line 9 of `/mnt/scratch/fast0/yiran/BLIP/sgb/Scene-Graph-Benchmark.pytorch/maskrcnn_benchmark/config/paths_catalog.py` to:
```
DATA_DIR = "/mnt/scratch/fast0/yiran/BLIP/sgb/Scene-Graph-Benchmark.pytorch/datasets/"
```
- Remember to download the category dataset specified in `VG_stanford_filtered_with_attribute` of Line 115 in the `paths_catalog.py` file. Specifically, need to download these two: 
```
"roidb_file": "vg/VG-SGG-with-attri.h5",
"dict_file": "vg/VG-SGG-dicts-with-attri.json",
```
3. Last checkpoint path in `/mnt/scratch/fast0/yiran/BLIP/Data/causal-motifs-sgdet/last_checkpoint`: change to 
`/mnt/scratch/fast0/yiran/BLIP/Data/causal-motifs-sgdet/model_0028000.pth`
