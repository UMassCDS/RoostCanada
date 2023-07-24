
# RoostCanada
This repo implements a machine learning system for detecting and tracking roosts 
in weather surveillance radar data.
Roost detection is based on [Detectron2](https://github.com/darkecology/detectron2) using PyTorch.

## Repository Overview
- **checkpoints** is for trained model checkpoints
- **development** is for developing detection models
- **src** is for system implementation
    - **data**
        - **downloader** downloads radar scans based on station and day; 
        scan keys and directories for downloaded scans are based on UTC dates
        - **renderer** renders numpy arrays from downloaded scans, visualizes arrays, 
        and deletes the scans after rendering; 
        directories for rendered arrays and images are based on UTC dates
    - **detection**
    - **evaluation** contains customized evaluation adapted from pycocotools v2.0.2
    - **tracking**
    - **utils** contains various utils, scripts to postprocess roost tracks, and scripts to generate visualization
- **tools** is for system deployment
    - **demo.py** downloads radar scans, renders arrays, detects and tracks 
    roosts in them, and postprocesses the results 
    - **launch_demo.py** is a modifiable template that submits **demo.sbatch** to servers with slurm management
    - **demo.ipynb** is for interactively running the system
    - **utc_to_local_time.py** takes in web ui files and append local time to each line

## Requirements
- See Detectron2 requirements
[here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).
- Find a compatible PyTorch version
[here](https://pytorch.org/get-started/previous-versions/).
## Installation
### GPU
1. To run detection with GPU, first check the cuda version at `/usr/local/cuda` or `nvcc -V`. 
2. Enter the following commands:
    ```bash
    conda create -n roostsys python=3.8
    conda activate roostsys
    
    # for development and inference with gpus, use the gpu version of torch; we assume cuda 11.3 here
    conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.3 -c pytorch -c conda-forge
    # for inference with cpus, use the cpu version of torch
    # conda install pytorch==1.10.0 torchvision==0.11.0 cpuonly -c pytorch
    
    git clone https://github.com/darkecology/roost-system.git
    cd roost-system
    pip install -e .
   ```
    - Note: If using the CPU version of PyTorch, ensure that conda can find the correct version of PyTorch. Run `conda search <lib>` and install the next available version that is compatible with the channel. 
    - Note: When running `pip install -e .` , ensure that pip can identify all dependencies, including open-cv2. If necessary, change the version of open-cv2 to a version that pip can identify. 
### Jupyter notebook
 1. Add the python environment to Jupyter:
    ```bash
    pip install jupyter
    conda install -c anaconda ipykernel
    python -m ipykernel install --user --name=roostsys
    ```
2.  Run the following commands to check which environments are in Jupyter as kernels or to delete one:
    ```bash
    jupyter kernelspec list
    jupyter kernelspec uninstall roostsys
    ```
3. Use the following command to run Jupyter notebook on a server:
    ```bash
    jupyter notebook --no-browser --port=9991
    ```
4. Use the following command to monitor from local: 
    ```bash
    ssh -N -f -L localhost:9990:localhost:9991 username@server
    ```
5. Enter `localhost:9990` from a local browser tab.

## Developing a detection model
- **development** contains all training and evaluation scripts.
- To prepare a training dataset (i.e. rendering arrays from radar scans and 
generating json files to define datasets with annotations), refer to 
**Installation** and **Dataset Preparation** in the README of 
[wsrdata](https://github.com/darkecology/wsrdata.git).
- Before training, optionally run **try_load_arrays.py** to make sure there's no broken npz files.

### Running inference
A Colab notebook for running small-scale inference can be found 
[here](https://colab.research.google.com/drive/1UD6qtDSAzFRUDttqsUGRhwNwS0O4jGaY?usp=sharing).
Large-scale deployment can be run on CPU servers as follows:
1. Under **checkpoints**, download a trained detection checkpoint.

2. [Configure AWS](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html) by
`aws configure`
in order to download radar scans. 
Enter `AWS Access Key ID` and `AWS Secret Access Key` as prompted. Enter
`us-east-1` for `Default region name` and nothing for `Default output format`.
Review the updated AWS config.
    ```bash
    vim ~/.aws/credentials
    vim ~/.aws/config
    ```

3. Modify **demo.py** for system customization. 
For example, DET_CFG can be changed to adopt a new detector.

4. Make sure the environment is activated. Then consider two deployment scenarios.
   1. In the first, we process consecutive days at stations (we launch one job for 
   each set of continuous days at a station). 
         - Modify VARIABLES in **tools/launch_demo.py**.
        - Under **tools**, run `python launch_demo.py` 
            to submit jobs to slurm and process multiple batches of data. 
   2. In the second, we process scattered days at stations (we launch one job for 
   all days from each station). 
        - Modify VARIABLES in **tools/gen_deploy_station_days_scripts.py**. 
        - Under **tools**, run `python gen_deploy_station_days_scripts.py` and then 
   `bash scripts/launch_deploy_station_days_scripts.sh`.   
        - Note: Each output txt file save scans or tracks 
   for one station-day. You need to manually combine txt files for station-days from each same station.

   3. GOTCHA 1: EXPERIMENT_NAME needs to be carefully chosen; 
  it will correspond to the dataset name later used in the web UI.
   
   4. GOTCHA 2: If there are previous batches processed under this EXPERIMENT_NAME 
   (i.e. dataset to be loaded to the website), we can move previously processed data in 
   the output directory to another location before saving newly processed data to this 
   EXPERIMENT_NAME output directory. When we copy newly processed data to the server 
   that hosts the web UI, previous data won't need to be copied again.

## Deploying locally
Roost data can be accessed and visualized locally without slurm by making a few changes to your configurations. 

### Generating the inference
1. Within * insert local file *, set the following variables based on your dataset:
    - `station`
    - `start`
    - `end`
    - `sun_activity`
    - `min_before`
    - `min_after`
2. Run the code, then open results in the <b>inference_results</b> folder. 
### Changing paths to local
1. Clone the [roostui](https://github.com/UMassCDS/roostui) repo to your system. 
2. Copy the <b>inference_results</b> folder and paste it into the root of the roostui repo. 
3. Make a new folder (<b>new_fldr</b>) within the <b>data</b> folder of the roostui repo. 
4. Navigate to the <b>scans_and_tracks</b> folder within the roost-system repo. Copy the scans and tracks files. 
5. Navigate to <b>new_fldr</b> within the roostui repo. Paste the scans and tracks files into your folder.
6. In your command line, run the following commands:
    ```bash 
    cd roostui
    cd data
    bash init_dataset.sh new_fldr
    ``` 
7. Open the newly created config.json file in <b>new_fldr</b>. This contains the reflectivity (dz) and velocity (vr) paths. 
8. Replace the doppler path with the following local path for reflectivity and velocity: inference_results/ui/img/. 
    - For example: 
        ``` bash
        #old path:
        "https://doppler.cs.umass.edu/roost/img/%s/dz05/%04d/%02d/%02d/%s/%s.png"
        #new path:
        "inference_results/ui/img/dz05/%04d/%02d/%02d/%s/%s.png"
        ```
9. Delete the "dataset" variable within "fields" for both "dz" and "vr."
### Visualizing the inference
1. Navigate to roostui/data/config.json. Open the config.json file.
2. Add the <b>new_fldr</b> to the list of datasets.
3. Run the following commands:
    ```bash
    cd roostui 
    yarn run build 
    yarn run serve
    ```
4. Open the outputted server link (it should have the local port number 8888). 
5. Find <b>new_fldr</b> in the Dataset dropdown menu and enter your date and time. An inference appears in the UI. 







