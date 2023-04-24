Installation
============
voxTrace can be installed/run in various ways on different systems. To optimize your general workflow when
working on the quantification of confocal measurements i reccomend one of the following two ways:

* Install locally on Windows machine using WSL and Ubuntu 20.04. An advantage of this is that you can use the OASYS GUI on Windows and generate the shadow files which you can use directly for.
* Install on a LINUX machine with high computing power (preferably CUDA support) and setup as a remote system.


A short statement by the author, concerning licencing: 
-------------------------------------------------------
This codes is supposed to be free to use in a way that you can basically use it
in any way you want, without any warranty from my side. I therefore chose the :ref:`licence-label`
MIT Licence. Nevertheless, before using/redistributing this code in 
a commercial way you should notice that some of the packages this code naturally 
depends on have different licences:

* `Armadillo`_: a C++ library for linear algebra & scientific computing                                (Apache 2.0 License)
* `OptimLib`_: a lightweight C++ library of numerical optimization methods for nonlinear functions.    (Apache 2.0 License)
* `Shadow3`_: an open source ray tracing code for modeling optical systems                             (MIT Licence)
* `SciPlot`_: a C++ scientific plotiing library powerd by gnupolot                                     (MIT License)

* `XrayLib`_: a library for interactions of X-rays with matter                                         (Special License -> see XRayLibAPI.hpp)
* `PolyCap`_: a C library to calculate X-ray transmission through polycapillaries                      (GNU General Public Licence)

* `Ensmallen`_: a flexible C++ library for efficient numerical optimization                            (3-clause BSD licence)
* `GSL`_ - GNU Scientific Library                                                                      (mostly GNU General Public Licence)


Dependencies
-------------

Ubuntu 20.04
~~~~~~~~~~~~~~~~~~
    -> shadow3 
        clone repo from https://github.com/PaNOSC-ViNYL/shadow3/
        switch to the the branch gfortran8-fixes (in the directory type git checkout gfortran8-fixes)
        

    -> xraylib from https://github.com/tschoonj/xraylib/wiki
        add to /etc/bash.bashrc : 
            export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
            export LD_LIBRARY_PATH=/usr/local/lib/

    -> optimlib from https://github.com/kthohr/optim (https://www.kthohr.com/optimlib_docs_de.html)
        ->armadillo from http://arma.sourceforge.net/ for dependencies using ArchLinux see https://aur.archlinux.org/packages/armadillo/
            -> arpack (sudo pacman -S arpack)
            -> blas / openblas
            -> lapack
            -> superlu
            -> cmake
            -> hdf5
            -> intel-mkl

    -> gsl
    
Manjaro (https://manjaro.org/) -> ArchLinux based OS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    -> git
    -> base-devel (all: gcc, make, pkgconf)
    -> gcc-fortran (bash command is still gfortran)
    -> cmake

.. _Armadillo: https://de.wikipedia.org
.. _OptimLib: https://de.wikipedia.org
.. _Shadow3: https://de.wikipedia.org
.. _SciPlot: https://de.wikipedia.org

.. _XrayLib: https://de.wikipedia.org
.. _PolyCap: https://de.wikipedia.org
.. _Ensmallen: https://de.wikipedia.org
.. _GSL: https://de.wikipedia.org