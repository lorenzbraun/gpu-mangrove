# GPU Mangrove
GPU Mangrove is a predictive model for execution time and power consumption of CUDA kernels. Details can be found in the  publication "A Simple Model for Portable and Fast Prediction of Execution Time and Power Consumption of GPU Kernels".

Due to size restrictions databases are not included in this repository. Execute `download_dbs.sh` to initially download and extract them.

## Requirements

We provide training data in this repository. To gather your own training data you need to use CUDA Flux.

https://github.com/UniHD-CEG/cuda-flux/

If you only want to train the model with our datasets python3 with tho following packages is recommended:

* numpy >= 1.18.1
* pandas >= 1.0.3
* pickle >= 4.0
* tqdm >= 4.45.0
* sklearn (scikit-learn) >= 0.22.1

git lfs is required to checkout the databases (https://git-lfs.github.com/)

## Usage

To simply build the models simply use the default target of the makefile. 
There is also a target for the leave-one-out training - `make loo`. 
FYI: The training can take up to multiple days.

Of course gpu mangrove can be used directly:

```
usage: mangrove.py [-h]
                   {process,filter,cv,llo,ablation,timemodel,paramstats} ...

GPU Mangrove: preprocess and filter datesets, train GPU Mangrove model with
cross-validation or leave-one-out, prediction with trained model.

positional arguments:
  {process,filter,cv,llo,ablation,timemodel,paramstats}
    process             Join features and measurements, apply feature
                        engineering
    filter              Limit the amunt of samples which are being user per
                        bench,app,dataset,kernel tuple
    cv                  train model using cross-validation
    llo                 train model using leave-one-out
    timemodel           measure prediction latency

optional arguments:
  -h, --help            show this help message and exit
```

**Examples:**

```
./mangrove.py process --fdb data/FeatureTable-0.3.db --mdb data/KernelTime-K20.db -o data/time_samples_K20_median.db
./mangrove.py filter -i data/time_samples_K20_median.db -t 100 -o data/time_samplesf_K20_median_100.pkl
./mangrove.py cv -i data/time_samplesf_K20_median_100.pkl -o data/time_model_K20_median_100.pkl -r data/time_cv-res_K20_median_100.pkl -t 30 -s 3 -k 5
```

## Jupyter Notebooks

* The visualization notebook is used to create plots using our sample data and the models. It can be used to examine the data and the model in depth.
* The FeatureProcessing notebook is used to create the FeatureTable database. It is not needed to execute it as this database is already there.

## Databases

The data folder contain databases which can be examined with sqlite3:

* CUDAFlux-0.3-Metrics.db - raw CUDA Flux metrics
* FeatureTable-0.3.db - processed CUDA FLux metrics
* KernelTime-$GPU.db - raw time measurements
* KernelPower-$GPU.db - raw power measurements

Entrys in FeatureTable and KernelTime/KernelPower are usually joined by the tuple (bench,app,dataset,lseq) where lseq stands for launch sequence and is the sequential order of the kernels launched in the application.
