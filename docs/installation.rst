Installation
============
voxTrace is a CUDA C++ package that can theoretically be run on 
various plattforms after modification of the MAKEFILE. voxTrace was 
developed and tested on Ubuntu 20.04 and 22.04. as well as almalinux on a HPC.
Depending on whether you have admin rights or not the installation varies slightly. 
For ease of use the option HPC=true can be set to store multiple paths for your 
local machine and your HPC without changing the MAKEFILE. Default for HPC is false.

A reproducible code ocean capsule can be found

Linux
------
This installation assumes a fresh install of Ubuntu 20.04 or Ubuntu 22.04. with admin rights.

* Run the script pre-install.sh to install all dependencies or install them manually
* Install `CUDA 12.1`_ 
* Run make fast to compile the code
* Create the dirs ./test-data/simulation/nist-1107/post-sample and ./test-data/simulation/nist-1107/detector
* Run the SampleTracer with run ./build/src/SampleTracer ./test-data/simulation/nist-1107
* Run the CapillaryTracer with run ./build/src/CapillaryTracer ./test-data/simulation/nist-1107

Modify the Capillaries.txt, Sample.txt, Simulation.txt and Polycapillary.txt to your needs. 
You can use the Jupyter-UI to create your own Materials.txt. 
For further assistance please write to michael.iro@tuwien.ac.at   

A short statement by the author, concerning licencing: 
-------------------------------------------------------
This codes is supposed to be free to use, without any warranty from my side. 
I therefore chose the `MIT Licence`_. Nevertheless, before 
using/redistributing this code in a commercial way you should notice that some 
of the packages this code naturally depends on have different licences:

* `Armadillo`_: a C++ library for linear algebra & scientific computing
* `Ensmallen`_: a flexible C++ library for efficient numerical optimization
* `GSL`_ - GNU Scientific Library (for flobal optimization algorithms)
* `SciPlot`_: a C++ scientific plotiing library powerd by gnupolot
* `XrayLib`_: a library for interactions of X-rays with matter
* `Shadow3`_: an open source ray tracing code for modeling optical systems
* `PolyCap`_: a C library to calculate X-ray transmission through polycapillaries

.. _Armadillo: https://arma.sourceforge.net/
.. _Ensmallen: https://ensmallen.org/
.. _GSL: https://www.gnu.org/software/gsl/
.. _SciPlot: https://sciplot.github.io/
.. _XrayLib: https://github.com/tschoonj/xraylib/wiki
.. _Shadow3: https://github.com/oasys-kit/shadow3
.. _PolyCap: https://github.com/PieterTack/polycap
.. _CUDA 12.1: https://developer.nvidia.com/cuda-downloads
.. _MIT Licence: https://michaeliro.github.io/voxTrace/licence.html