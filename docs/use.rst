How to use voxTrace
====================

voxTrace reads all parameters for the setup and the simulation from 5 txt-files. The location of the folder of these files should be given to the executables as a path. In this folder, there should be two folders named post-sample and detector. The files are (with example values):

Capillaries: Everything in um and °
-----------------------------------

These parameters provide specific details about the capillaries' sizes, positions, and their relationships to each other and the detector.

Primary Capillary Geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Radius: 1075.0 um - Radius of the Primary Capillary exit window
- Initial Distance to Detector: 5100.0 um - Initial distance from the Primary Capillary exit window to the sample
- Single Capillary Diameter: 16.5 um - Diameter of a single Capillary of the Primary Capillary
- Energy: 17.4 keV - Energy of the X-rays

Primary Capillary Transformation Parameters (Position)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- X: 150.0 - X-Position of the Primary Capillary
- Y: 150.0 - Y-Position of the Primary Capillary
- Z: 0.0 - Z-Position of the Primary Capillary
- Distance to Detector: 5100.0 um - Focal distance of the Primary Capillary
- Angle: 45.0° - Angle of the Primary Capillary to the sample surface

Secondary Capillary Transformation Parameters (Position)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- X: 150.0 - X-Position of the Secondary Capillary
- Y: 150.0 - Y-Position of the Secondary Capillary
- Z: 0.0 - Z-Position of the Secondary Capillary
- Distance to Detector: 4900.0 um - Distance from the sample to the Secondary Capillary input window
- Angle: 45.0° - Angle of the Secondary Capillary to the sample surface
- Radius of Input window of the secondary capillary: 950.0 um - Radius of the Secondary Capillary input window

Sample parameters
-----------------

These parameters provide specific details about the sample's size, position, and type.

Start Position
~~~~~~~~~~~~~~

- X: 0.0 - X-Position of the sample
- Y: 0.0 - Y-Position of the sample
- Z: 0.0 - Z-Position of the sample

Sample Dimensions
~~~~~~~~~~~~~~~~~

- X: 300.0 - X-Size of the sample in um
- Y: 300.0 - Y-Size of the sample in um
- Z: 150.0 - Z-Size of the sample in um

Voxel Dimensions
~~~~~~~~~~~~~~~~

- X: 5.0 - X-Size of a voxel in um
- Y: 5.0 - Y-Size of a voxel in um
- Z: 5.0 - Z-Size of a voxel in um

Sample Type (0 = homogeneous, 1 = layered, 2 = heterogeneous)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Type: 0 - Type of the sample (for now not used in calculation only for overview)

Simulation parameters
---------------------

These parameters provide specific details about the simulation itself.

General Parameters
~~~~~~~~~~~~~~~~~~

- Number of Measurement Points: 11
- Number of Rays to Simulate per Measurement Point Hitting the Secondary Polycapillary Window: 30000

Scan-Path as Offset in x, y, z in um in Coordinate System of the Sample
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Measurement Point 1: 0.0 um, 0.0 um, -50.0 um
- Measurement Point 2: 0.0 um, 0.0 um, -40.0 um
- Measurement Point 3: 0.0 um, 0.0 um, -30.0 um
- Measurement Point 4: 0.0 um, 0.0 um, -20.0 um
- Measurement Point 5: 0.0 um, 0.0 um, -10.0 um
- Measurement Point 6: 0.0 um, 0.0 um, 0.0 um
- Measurement Point 7: 0.0 um, 0.0 um, 10.0 um
- Measurement Point 8: 0.0 um, 0.0 um, 20.0 um
- Measurement Point 9: 0.0 um, 0.0 um, 30.0 um
- Measurement Point 10: 0.0 um, 0.0 um, 40.0 um
- Measurement Point 11: 0.0 um, 0.0 um, 50.0 um

Materials
---------

These parameters provide specific details about the materials in each voxel of the sample.

This file can be generated using the following code:

.. code-block:: python

    import numpy as np

    # Define sample size and voxel size
    sample_size = np.array([600., 600., 200.])  # in units of micrometers
    voxel_size = np.array([5.0, 5.0, 5.0])  # in units of micrometers

    # Define materials Triple-Cross
    materials = [
        {
            "z_range": (0, 50),
            "elements": [6, 24],
            "mass_fractions": [0.999651886257307, 0.00034811374269286]
        },
        {
            "z_range": (50, 100),
            "elements": [6, 27],
            "mass_fractions": [0.999543658490524, 0.000456341509475796]
        },
        {
            "z_range": (100, 150),
            "elements": [6, 30],
            "mass_fractions": [0.999378760356556, 0.000621239643443875]
        },
        {
            "z_range": (150, 200),
            "elements": [6],
            "mass_fractions": [1.0]
        }
    ]

    # Write output to text file
    with open(path_start + "/Materials.txt", "w") as f:
        # Write header
        f.write("Materials\n")
        f.write("=========\n\n")

        for i in range(np.prod(num_voxels)):
            # Write point information
            x, y, z = np.unravel_index(i, num_voxels)
            for material in materials:
                if material["z_range"][0] / voxel_size[2] <= z < material["z_range"][1] / voxel_size[2]:
                    f.write("\nPoint {}\n".format(i + 1))
                    f.write("-" * 80 + "\n\n")
                    f.write("Coordinates (x, y, z): {}, {}, {}\n\n".format(x, y, z))
                    f.write("Number of Elements: {}\n\n".format(len(material["elements"])))
                    f.write("  Elements Z: {}\n\n".format(", ".join(map(str, material["elements"]))))
                    f.write("Element Mass Fractions: {}\n\n".format(", ".join(map(str, material["mass_fractions"]))))
                    break

    path_start = "/media/miro/Data-1TB/simulation/triple-cross"

    # Calculate number of voxels in each dimension
    num_voxels = np.ceil(sample_size / voxel_size).astype(int)


Point 1
~~~~~~~

- Coordinates (x, y, z): 0, 0, 0
- Number of Elements: 6
- Elements Z: 26, 28, 29, 30, 50, 82
- Element Mass Fractions: 0.0004, 0.001, 0.6119, 0.3741, 0.0107, 0.0019

Point 2
~~~~~~~

- Coordinates (x, y, z): 0, 0, 1
- Number of Elements: 6
- Elements Z: 26, 28, 29, 30, 50, 82
- Element Mass Fractions: 0.0004, 0.001, 0.6119, 0.3741, 0.0107, 0.0019

Point 3
~~~~~~~

- Coordinates (x, y, z): 0, 0, 2
- Number of Elements: 6
- Elements Z: 26, 28, 29, 30, 50, 82
- Element Mass Fractions: 0.0004, 0.001, 0.6119, 0.3741, 0.0107, 0.0019

Point 4
~~~~~~~

- Coordinates (x, y, z): 0, 0, 3
- Number of Elements: 6
- Elements Z: 26, 28, 29, 30, 50, 82
- Element Mass Fractions: 0.0004, 0.001, 0.6119, 0.3741, 0.0107, 0.0019

Point 5
~~~~~~~

- Coordinates (x, y, z): 0, 0, 4
- Number of Elements: 6
- Elements Z: 26, 28, 29, 30, 50, 82
- Element Mass Fractions: 0.0004, 0.001, 0.6119, 0.3741, 0.0107, 0.0019
