2. Installation
===============

Installing with pip (preferred)
-----------------------------------

If you have Python3 and pip installed on your machine, then the VARS-TOOL package can be installed as following::

    foo@bar:~$  pip install varstool


Installing from source code
-------------------------------

To install the lastest VARS-TOOL code from the source code, you will need to clone the github repository onto your local device using the command::

    foo@bar:~$ git clone https://github.com/vars-tool/vars-tool.git

To install the package, enter the VARS-TOOL directory and run::

    foo@bar:~$ cd vars-tool
    foo@bar:~$ pip install .

If pip is not available on your device use::

    foo@bar:~$ python setup.py install

Note: if installation does not work due to limited permissions, add the ``--user`` option to the install commands.
