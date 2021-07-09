# VARS-TOOL Python Library

## 1. Overview of VARS-TOOL

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


## 2. Installation

### 2.1. Installing with pip

If you have Python3 and pip installed on your machine, then the vars-tool package can be installed using pip:
```console
foo@bar:~$  pip install varstool
```

### 2.2. Installing from source code

To install vars-tool from the source code, you will need to clone the github repository onto your local device using the command:
```console
foo@bar:~$ git clone https://github.com/vars-tool/vars-tool.git
```
To install the package, enter the vars-tool directory and run:
```console
foo@bar:~$ cd vars-tool
foo@bar:~$ pip install .
```
If pip is not available on your device use:
```console
foo@bar:~$ python setup.py install
```
> **_Note:_**: If this does not work due to limited permissions, add the `--user` option to the above commands.

## 3. Documentation

Under progres...

## 4. Examples and Tutorials

* [4.1. Quick Start:](https://github.com/vars-tool/vars-tool/blob/master/tutorial/QucikStart-Tutorial.ipynb) Here is an overview of the varstool main components and a simple example using ishigami and wavy6d models.

* [4.2. Real-world Example Using HBV-SASK (using TSVARS):](url) Sensitivity analysis of a real-world hydrological model 
	
* [4.3. Handling Correlated Factors Using Generalized VARS (GVARS):](url) under progress...
	
* [4.4. Data-driven Sensitivity Analysis (DVARS):](url) under progress...

## Your Contributions

Contributions to vars-tool are welcome! To contribute new features submit a pull request. To get started it is recommended to install the packages in `requirements.txt` by using the following command:
```console
foo@bar:~$ pip install -r requirements.txt
```
Once the packages are installed to contribute do the following:
1. Fork the repository ([here](https://github.com/vars-tool/vars-tool/fork)). A fork makes it possible to make changes to the source code through creating a copy,
2. Create a new branch on your fork,
3. Commit your changes and push them to your branch, and
4. When the branch is ready to be merged, you can create a Pull Request ([how to create a pull request](https://gist.github.com/MarcDiethelm/7303312)).

## References

## License


