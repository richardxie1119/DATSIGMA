# DATSIGMA
**DA**ta-driven **T**ools for **S**ingle-cell analysis using **I**mage-**G**uided **MA**ss spectrometry

<p >
  <img src="https://github.com/richardxie1119/DATSIGMA/blob/main/DATSIGMA_logo.png" /width="450"> 
</p>

## What's included
This is the code repository containing the data-driven and machine learning based framework for image-guided single-cell MS data processing and interpretation.


## Dependencies
- Numpy >=1.19.2
- Scipy >=1.6.2
- sklearn >=1.1.2 
- PyTorch ==1.4.0
- Spike (FTMS processing): https://github.com/spike-project/spike

## Installation via Anaconda (recommended)
1. First nevigate into the directory: `cd DATSIGMA`
2. Create conda virtual env: `conda env create -f environment.yml`
3. Activate virtual env: `conda activate datsigma`
4. Install Jupyter Notebook: `conda install -c anaconda ipykernel`
5. Add virtual env to kernel: `python -m ipykernel install --user --name=datsigma`

The repository contains:
- [ ] Signal, image, and MS data preprocessing modules.
- [ ] Unsupervised analysis modules.
- [ ] Machine learning modules.

## Interactive demos
- [processing 30,000 single cells raw high-resolution MS data](https://github.com/richardxie1119/DATSIGMA/blob/main/tutorial/reanalysis_30k_raw.ipynb)
- [exploratory analysis of neurons coupled with immunostaining](https://github.com/richardxie1119/DATSIGMA/blob/main/tutorial/scms_immunostain.ipynb)
- [machine learning classification and feature selection for neurons vs. astrocytes](https://github.com/richardxie1119/DATSIGMA/blob/main/tutorial/ICC_neuron_vs_astro.ipynb)
- [single vesicle classifications](https://github.com/richardxie1119/DATSIGMA/blob/main/tutorial/vesicle_classification.ipynb)
- [analysis of cells of developing brain](https://github.com/richardxie1119/DATSIGMA/blob/main/tutorial/developing_brain.ipynb)
- [20,000 single aplysia neurons from six types of ganglia](https://github.com/richardxie1119/DATSIGMA/blob/main/tutorial/supervised_aplysia.ipynb)

## Data availability
Raw high-resolution FTMS data are available upon request due to large size. Processed data sets are available at:
