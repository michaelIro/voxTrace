Installation
============
voxTrace is a CUDA C++ package that can theoretically be run on various plattforms after 
modification of the MAKEFILE. voxTrace was developed and tested on Ubuntu 20.04 and 22.04.

Ubuntu 
-------
*  Start with 

A short statement by the author, concerning licencing: 
-------------------------------------------------------
This codes is supposed to be free to use, without any warranty from my side. 
I therefore chose the :ref:`licence-label`MIT Licence. Nevertheless, before 
using/redistributing this code in a commercial way you should notice that some 
of the packages this code naturally depends on have different licences:

* `Armadillo`_: a C++ library for linear algebra & scientific computing                                (Apache 2.0 License)
* `OptimLib`_: a lightweight C++ library of numerical optimization methods for nonlinear functions.    (Apache 2.0 License)
* `Shadow3`_: an open source ray tracing code for modeling optical systems                             (MIT Licence)
* `SciPlot`_: a C++ scientific plotiing library powerd by gnupolot                                     (MIT License)

* `XrayLib`_: a library for interactions of X-rays with matter                                         (Special License -> see XRayLibAPI.hpp)
* `PolyCap`_: a C library to calculate X-ray transmission through polycapillaries                      (GNU General Public Licence)

* `Ensmallen`_: a flexible C++ library for efficient numerical optimization                            (3-clause BSD licence)
* `GSL`_ - GNU Scientific Library                                                                      (mostly GNU General Public Licence)

.. _Armadillo: https://de.wikipedia.org
.. _OptimLib: https://de.wikipedia.org
.. _Shadow3: https://de.wikipedia.org
.. _SciPlot: https://de.wikipedia.org

.. _XrayLib: https://de.wikipedia.org
.. _PolyCap: https://de.wikipedia.org
.. _Ensmallen: https://de.wikipedia.org
.. _GSL: https://de.wikipedia.org