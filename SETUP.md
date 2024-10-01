All analyses were done using IPython notebooks, partially in R 4.0.0 or higher, overall, in Python 3.10; **to classify a sample, one only needs Python 3.10.**  

### Windows users
Install WSL, following the instructions provided on the [WSL installation webpage](https://learn.microsoft.com/en-us/windows/wsl/install).

### Prepare Python environment
Use terminal for the next steps (WSL, Unix-based or macOS).

Install python if needed:
```bash
sudo apt-get update 
sudo apt-get install python3.10-venv python3.10-dev python3-pip
```
Clone repo and run script to install required packages:

```bash
git clone git@github.com:BostonGene/Immune_Escape.git
bash make_tme_environment.sh
```

The script will create python3.10 environment with all necessary packages and ipykernel core named "ImmEsc_venv" for it. We recommend to use this kernel further.

#### Conda
We don't recommend setting up the environment with conda and requirements.txt aren't adapted for conda. However, one can follow this [link](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/installing-with-conda.html) to install packages; it could be necessary to use conda-forge for missing packages. Conda-environment also should be added as Jupyter kernel:

```bash
python -m ipykernel install --user --name=ImmEsc_venv
```

### R
We used R tools in the paper ([ComBat](https://rdrr.io/bioc/sva/man/ComBat.html) and [limma](https://www.bioconductor.org/packages/release/bioc/html/limma.html)). Install R 4.0.0 or higher via [CRAN](https://cran.r-project.org/). Unix and macOS users can follow installation via [rig](https://github.com/r-lib/rig). To install required libraries, in R terminal write the script:

```R
cran_packages <- c("tidyverse", "BiocManager")

# install CRAN packages if needed
for (pkg in cran_packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        message(paste("Installing", pkg, "from CRAN..."))
        install.packages(pkg)
    } else {
        message(paste(pkg, "is already installed."))
    }
}
bioc_packages <- c("sva", "limma")

# install Bioconductor packages if needed
for (pkg in bioc_packages) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        message(paste("Installing Bioconductor package:", pkg, "..."))
        BiocManager::install(pkg)
    } else {
        message(paste(pkg, "is already installed."))
    }
}
```
