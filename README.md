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

### 2.1. Installing with pip (preferred)

If you have Python3 and pip installed on your machine, then the VARS-TOOL package can be installed as following:
```console
foo@bar:~$  pip install varstool
```


### 2.2. Installing from source code

To install the lastest VARS-TOOL code from the source code, you will need to clone the github repository onto your local device using the command:
```console
foo@bar:~$ git clone https://github.com/vars-tool/vars-tool.git
```
To install the package, enter the VARS-TOOL directory and run:
```console
foo@bar:~$ cd vars-tool
foo@bar:~$ pip install .
```
If pip is not available on your device use:
```console
foo@bar:~$ python setup.py install
```
| :point_up:    | If installation does not work due to limited permissions, add the `--user` option to the install commands.|
|---------------|:----------------------------------------------------------------------------------------------------------|


## 3. Documentation

Under progress...


## 4. Examples and Tutorials

* 4.1. **Quick Start**: Here is an overview of the VARS-TOOL main components and a simple example using ishigami and wavy6d models. [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/vars-tool/vars-tool/master?filepath=tutorial%2FQucikStart-Tutorial.ipynb)

* 4.2. **Real-world Example Using HBV-SASK (using TSVARS)**: Sensitivity analysis of a real-world hydrological model. [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/vars-tool/vars-tool/master?filepath=%2Ftutorial%2FTSVARS-Tutorial.ipynb)
	
* 4.3. **Handling Correlated Factors Using Generalized VARS (GVARS)**: under progress...
	
* 4.4. **Data-driven Sensitivity Analysis (DVARS)**: under progress...


## 5. Your Contributions

Contributions to VARS-TOOL are welcome! To contribute new features submit a pull request. To get started it is recommended to install the packages in `requirements.txt` by using the following command:
```console
foo@bar:~$ pip install -r requirements.txt
```
Once the packages are installed to contribute do the following:
1. Fork the repository ([here](https://github.com/vars-tool/vars-tool/fork)). A fork makes it possible to make changes to the source code through creating a copy,
2. Create a new branch on your fork,
3. Commit your changes and push them to your branch, and
4. When the branch is ready to be merged, you can create a Pull Request ([how to create a pull request](https://gist.github.com/MarcDiethelm/7303312)).


## 6. References

Under progress...


## 7. License

### 7.1 Software

VARS-TOOL is licensed under the GNU General Public License, Version 3.0 or later.

Copyright (C) 2015-21 Saman Razavi, University of Saskatchewan

This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 1, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston MA 02110-1301 USA.

### 7.2 Documentation 

Copyright (C) 2015-21 Saman Razavi, University of Saskatchewan

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/3.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/3.0/88x31.png" /></a><br />This documentation is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/3.0/">Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License</a>.
