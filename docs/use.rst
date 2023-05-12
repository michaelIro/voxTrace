How to use voxTrace
====================
voxTrace reads all parameters for the setup and the simulation from 5 txt-files.
For the simulation to work properly these files should be located in a folder
with two subfolders named post-sample and detector. The location of the folder 
should be given to the executables as a path. 

The SampleTracer needs the following files:
- Capillaries 
- Sample 
- Simulation 
- Materials 

and writes to the post-sample folder.

The CapillaryTracer needs the following files:
- Polycapillary

and writes to the detector folder.

Capillaries
------------
These parameters provide specific details about the capillaries' sizes, positions, 
and their relationships to each other and the detector.

Primary Capillary Geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~
- Radius of the Primary Capillary exit window: 1075.0 μm
- Initial distance from the Primary Capillary exit window to the sample: 5100.0 μm
- Diameter of a single Capillary of the Primary Capillary: 16.5 μm
- Energy of the X-rays: 17.4 keV 

Primary Capillary Transformation Parameters (Position)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- X-Position of the Primary Capillary focal point: 150.0 μm
- Y-Position of the Primary Capillary focal point: 150.0 μm
- Z-Position of the Primary Capillary focal point: 0.0 μm
- Focal distance of the Primary Capillary: 5100.0 μm
- Angle between primary polycapillary optical axis and sample surface: 45.0°

Secondary Capillary Transformation Parameters (Position)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- X-Position of the Secondary Capillary focal point: 150.0 μm  
- Y-Position of the Secondary Capillary focal point: 150.0 μm
- Z-Position of the Secondary Capillary focal point: 0.0 μm 
- Distance from the sample to the Secondary Capillary input window: 4900.0 μm
- Angle between secondary polycapillary optical axis and sample surface: 45.0°
- Radius of the Secondary Capillary input window: 950.0 μm

Sample parameters
-----------------
These parameters provide specific details about the sample's size, position, and type.

Start Position
~~~~~~~~~~~~~~
- X-Position of the sample: 0.0 μm
- Y-Position of the sample: 0.0 μm
- Z-Position of the sample: 0.0 μm

Sample Dimensions
~~~~~~~~~~~~~~~~~
- X-Size of the sample: 300.0 μm
- Y-Size of the sample: 300.0 μm
- Z-Size of the sample: 150.0 μm

Voxel Dimensions
~~~~~~~~~~~~~~~~
- X-Size of a voxel: 5.0 μm 
- Y-Size of a voxel: 5.0 μm
- Z-Size of a voxel: 5.0 μm

Sample Type 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Type of the sample (for now not used in calculation only for overview): 0 = homogeneous, 1 = layered, 2 = heterogeneous

Simulation parameters
---------------------
These parameters provide specific details about the simulation itself.

General Parameters
~~~~~~~~~~~~~~~~~~
- Number of Measurement Points: 11
- Number of Rays to Simulate per Measurement Point Hitting the Secondary Polycapillary Window: 30000

Scan-Path as offset in x, y, z in μm in coordinate system of the sample
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Measurement Point 1: 0.0 μm, 0.0 μm, -50.0 μm
- Measurement Point 2: 0.0 μm, 0.0 μm, -40.0 μm
- Measurement Point 3: 0.0 μm, 0.0 μm, -30.0 μm
- Measurement Point 4: 0.0 μm, 0.0 μm, -20.0 μm
- Measurement Point 5: 0.0 μm, 0.0 μm, -10.0 μm
- Measurement Point 6: 0.0 μm, 0.0 μm, 0.0 μm
- Measurement Point 7: 0.0 μm, 0.0 μm, 10.0 μm
- Measurement Point 8: 0.0 μm, 0.0 μm, 20.0 μm
- Measurement Point 9: 0.0 μm, 0.0 μm, 30.0 μm
- Measurement Point 10: 0.0 μm, 0.0 μm, 40.0 μm
- Measurement Point 11: 0.0 μm, 0.0 μm, 50.0 μm

Materials
---------

These parameters provide specific details about the materials in each voxel of the sample.

This file can be generated using the following code:

.. code-block:: python

    import numpy as np

    # Define sample size and voxel size
    sample_size = np.array([600., 600., 200.])  # in units of micrometers
    voxel_size = np.array([5.0, 5.0, 5.0])  # in units of micrometers

    path_start = "/media/miro/Data-1TB/simulation/triple-cross"

    # Calculate number of voxels in each dimension
    num_voxels = np.ceil(sample_size / voxel_size).astype(int)

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


Polycapillary
--------------
- Optic Length: 4.03 cm              
- External Radius Upstream: 0.095 cm            
- External Radius Downstream: 0.3175 cm           
- Single Capillary Radius at Optic Entrance: 0.0000975 cm           
- Single Capillary Radius at Optic Exit: 0.000325 cm         
- Focal Distance on Entrance Window Side: 0.49 cm               
- Focal Distance on Exit Window Side: 100000000.0 cm         
- Amount of Elements in Optic Material: 2                   
- Polycapillary Optic Material Composition - Atomic Numbers: 8, 14                              |
- Polycapillary Optic Material Composition - Weight Percentages SiO2: 53.0, 47.0    
- Optic Material Density: 2.23 g/cm^3              
- Surface Roughness: 5.0 Angstrom             
- Number of Capillaries in the Optic: 240000           

