# voxTrace
A voxel based Monte-Carlo Ray tracing C++ code for quantitative confocal micro X-ray fluorescence analysis.

For a detailed documentation see ...

If you use this code for your scientific work please refer to this paper ... in your publications.

## A short statement by the author, concerning licencing: 
This codes is supposed to be free to use in a way that you can basically use it
in any way you want, without any warranty from my side. I therefore chose the 
MIT Licence (see [LICENSE](LICENSE)). Nevertheless, before using/redistributing this code in 
a commercial way you should notice that some of the packages this code naturally 
depends on have different licences:

    - Armadillo: a C++ library for linear algebra & scientific computing                                (Apache 2.0 license)
    - Ensmallen: a flexible C++ library for efficient numerical optimization                            (3-clause BSD licence)
    - SciPlot: a C++ scientific plotiing library powerd by gnupolot                                     (MIT License)

    - XrayLib: a library for interactions of X-rays with matter                                         (Special License -> see XRayLibAPI.hpp)
    - Shadow3: an open source ray tracing code for modeling optical systems                             (MIT Licence)
    - PolyCap: a C library to calculate X-ray transmission through polycapillaries                      (GNU General Public Licence)


    <!--- GSL - GNU Scientific Library                                                                      (mostly GNU General Public Licence)
    - OptimLib: a lightweight C++ library of numerical optimization methods for nonlinear functions.    (Apache 2.0) -->


## Installation
voxTrace can be installed/run in various ways on different systems. To optimize your general workflow when
working on the quantification of confocal measurements i reccomend one of the following two ways:

    -   Install locally on Windows machine using WSL and Ubuntu 20.04. An advantage of this is that you can use 
        the OASYS GUI on Windows and generate the shadow files which you can use directly for.
        
    -   Install on a LINUX machine with high computing power (preferably CUDA support) and setup as a remote system.

In both cases i would recommend setting up the nodejs-Web-GUI i made for this programm 

### Optional Instrucion Video
If you are new to Linux or have problems installing some of the dependency packages please follow the instructions 
below or watch this video i made with the instructions for setting up your system: https://github.com
