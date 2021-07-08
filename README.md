# VARS-TOOL Python Library

### Overview of VARS-TOOL
VARS-TOOL is a next-generation, multi-method sensitivity and uncertainty analysis software toolbox,
applicable to the full range of computer simulation models, including Dynamical Earth Systems
Models (DESMs). Developed primarily around the powerful “Variogram Analysis of Response
Surfaces” (VARS) framework, VARS-TOOL provides a comprehensive suite of algorithms and
methods for global sensitivity analysis (GSA), including ones based on the derivative-based (such as
the method of Morris), distribution-based (particularly the variance-based method of Sobol’), and
variogram-based (such as VARS) approaches.

The underlying approach of the VARS-TOOL toolkit is described in detail in the following publications:

1. A new framework for comprehensive, robust, and efficient global sensitivity analysis: 1. Theory: https://doi.org/10.1002/2015WR017558 
2. A new framework for comprehensive, robust, and efficient global sensitivity analysis: 2. Application: https://doi.org/10.1002/2015WR017559
3. VARS-TOOL: A toolbox for comprehensive, efficient, and robust sensitivity and uncertainty analysis: https://doi.org/10.1016/j.envsoft.2018.10.005


### Installation
#### Installing with pip
If you are using Linux or macOS the VARS-TOOL package can be installed using pip:
```
pip install varstool
```

#### Installing from source
First, you will need to clone the repository onto your local device using the command:
```
git clone https://github.com/vars-tool/vars-tool.git
```
To install the package, run:
```
pip install .
```
If pip is not available on your device use:
```
python setup.py install
```
note: if this does not work add the `--user` option to the above commands.

### Documentation

### Examples
#### [Quick Start](url)
Here is an overview of the varstool main components and a simple example using ishigami and wavy6d models.

#### [Real-world Example Using HBV-SASK (using TSVARS)](url)
	
#### [Handling Correlated Factors Using Generalized VARS (GVARS)](url)
	
#### [Data-driven Sensitivity Analysis (DVARS)](url)

### Contribution
Contributions to VARS-TOOl are welcome! To contribute new features submit a pull request. To get started it is recommended to install the packages in `requirements.txt` by using the following command:
```
pip install -r requirements.txt
```
Once the packages are installed to contribute do the following:
1. Fork the repository([here](https://github.com/vars-tool/vars-tool/fork)). A fork is a copy where you can make your changes to the source code.
2. Create a new branch on your fork
3. Commit your changes and push them to your branch
4. When the branch is ready to be merged, create a Pull Request ([how to create a pull request](https://gist.github.com/MarcDiethelm/7303312))

### References

### License


