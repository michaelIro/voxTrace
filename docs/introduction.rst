What is voxTrace?
===================
voxTrace is a voxel based Monte-Carlo Ray tracing CUDA C++ code for quantitative confocal micro X-ray fluorescence analysis. 
While the code can be used / be adapted to be used for different problems/setups, its main purpose, for which it has been 
tested, is the simulation of Energy dispersive Micro X-ray fluoprescence spectra in a confocal setup (CMXRF).

This works in 4 steps:

* Generate X-rays from a source and optionally trace them through optical elements (mirrors, multilayer, etc.) modelled in `Shadow3`_.
* Trace the generated X-rays through a primary polycapillary optic modelled in `polycap`_. 
* Simulate the X-ray matter interactions in the voxel based sample.
* Trace the generated X-rays through a secondary polycapillary optic modelled in `polycap`_. 

The first two steps can (and currently should) be skipped simulating the beam 
exitig the primary polycapillary exit window with a two-dimensional normal distribution. That this is a valid assumption is shown in this paper.

This vastly speeds up the process, since the primary beam can be generated on the GPU rather than by reading large files.
Current development of the software focuses on the implementation of global optimization algorithms for an iterative quantitative interpretation of CMXRF measurements and a more approachable User Interface.

.. _Shadow3: https://github.com/oasys-kit/shadow3 
.. _polycap: https://pietertack.github.io/polycap
.. _paper: https://de.wikipedia.org