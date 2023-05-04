# CS-766: Optimized Video Object Segmentation for implementation over FPGAs

## Introduction

In this project, we perform the task of video object segmentation over images taken from various sources such as traffic cameras for identifying objects (vehicles in our case) using effective deep learning models and then implement viable models on an FPGA simulation through VIVADO and metric against task key metrics (like accuracy, mean IoU, Jaccard Score, etc) and constraint performance metrics like resource consumption (e.g. latency, DSP, LUT).

# Repository:
- Modules: Contain standalone files containing codes for various parts of the project separately
- Notebooks: The end-to-end process implemented and visualized through ipynb files
- See HTML in the right
- Documents: The presentations, reports and any additional result files all in one place

## Setup Environment

Run the follwing code. For HLS4ML we will be using the latest version existing in the repository. Hence, we will install the library using git directly.

``` 
pip install -r requirements.txt
pip install git+https://github.com/fastmachinelearning/hls4ml.git
```

For VIVADO we will be using the 2020 version of the software which is compatible with the current build of the hls4ml library. It can be downloaded using the following link: https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vivado-design-tools/archive.html. Make sure to have Vivado HLS as part of the installation.

### Setting up data
For the project we have used vehicle based subset of the DAVIS 2016 dataset. You may access the data using the follwing link: https://davischallenge.org/davis2016/code.html

Following this you may setup the data in the following way
- Create two folders: images and masks
- Distribute the image data from the dataset downloaded between the two folders

### Post Project Compilation

After compiling the project using hls4ml, cd to the project folder generated and run the following code in your terminal/CLI.

``` 
nohup vivado_hls -f build_prj.tcl "reset=1 synth=1 csim=0 cosim=0 validation=0 export=0 vsynth=0" > vhls_synth.out &
tail -f vhls_synth.out
``` 

After the process is successfully executed the complete synthesis report is made available for analysis at:

``` 
your_hls_project_folder/myproject_prj/solution1/syn/report/myproject_csynth.rpt
``` 