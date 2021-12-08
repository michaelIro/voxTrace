What is voxTrace?
===================
voxTrace is a voxel based Monte-Carlo Ray tracing C++ code for quantitative confocal micro X-ray fluorescence analysis. While the code can be used / be adapted to be used for different problems/setups, its main purpose, for which it has been tested (see this `paper`_), is the quantification of Energy dispersive Micro X-ray fluoprescence meaurements in a confocal setup (CMXRF).

This works in 5 steps:

* Generate X-rays from a source and optionally trace them through optical elements (mirrors, multilayer, etc.) modelled in `Shadow3`_.
* Trace the generated X-rays through a primary polycapillary optic modelled in `polycap`_. 
* Main voxTrace Part
* Trace the generated X-rays through a secondary polycapillary optic modelled in `polycap`_. 
* Detector

This is a paragraph that contains 

.. _Shadow3: https://github.com/oasys-kit/shadow3 
.. _polycap: https://pietertack.github.io/polycap
.. _paper: https://de.wikipedia.org