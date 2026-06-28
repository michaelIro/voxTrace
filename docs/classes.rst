API reference
=============

The physics core is header-only (``src/core``); the ``*API`` classes
(``src/api``) wrap external libraries. The descriptions below are extracted
from the in-source Doxygen comments by Breathe, so they stay in sync with the
code. See :doc:`architecture` for how these classes fit together.

Core — the ray and its operators
--------------------------------

.. doxygenclass:: Ray
   :members:

.. doxygenclass:: SourceBase
   :members:

.. doxygenclass:: Source
   :members:

.. doxygenclass:: XRayTube
   :members:

.. doxygenclass:: Synchrotron
   :members:

.. doxygenclass:: LiquidMetalJet
   :members:

.. doxygenclass:: Sample
   :members:

.. doxygenclass:: Voxel
   :members:

.. doxygenclass:: Material
   :members:

.. doxygenclass:: ChemElement
   :members:

.. doxygenclass:: PolyCap
   :members:

.. doxygenclass:: Detector
   :members:

.. doxygenclass:: Tracer
   :members:

.. doxygenstruct:: RNG
   :members:

Core — output
-------------

.. doxygenclass:: Spectrum
   :members:

APIs to external software
-------------------------

.. doxygenclass:: XRayLibAPI
   :members:

.. doxygenclass:: OptimizerAPI
   :members:

.. doxygenclass:: PlotAPI
   :members:
