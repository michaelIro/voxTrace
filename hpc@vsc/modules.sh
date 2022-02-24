#!/bin/bash
module load gcc/10.2.0-gcc-9.1.0-2aa5hfe
spack load openblas@0.3.9
spack load openmpi@4.1.1%gcc@10.2.0
spack load /zymljkm #cmake 
spack load /mpxfn42 #hdf5

module load gcc/9.1.0-gcc-4.8.5-mj7s6dg

spack load armadillo@9.900.3%gcc@9.1.0
spack load atlas
spack load cmake@3.17.3%gcc@9.1.0

spack load /fd4ux3a #hdf5

export PKG_CONFIG_PATH=$HOME/Software/3rd-Party/Install/lib/pkgconfig
export PKG_CONFIG_PATH=$HOME/Software/3rd-Party/Install/xraylib/lib/pkgconfig:$HOME/Software/3rd-Party/Install/easyRNG/lib/pkgconfig

export HDF5_CFLAGS=/opt/sw/spack-0.12.1/opt/spack/linux-centos7-x86_64/gcc-9.1.0/hdf5-1.10.5-ltwm3rl6wginplgxx2p4ux44efi5qtek/include
export HDF5_LIBS=/opt/sw/spack-0.12.1/opt/spack/linux-centos7-x86_64/gcc-9.1.0/hdf5-1.10.5-ltwm3rl6wginplgxx2p4ux44efi5qtek/lib
export H5CC=/opt/sw/spack-0.12.1/opt/spack/linux-centos7-x86_64/gcc-9.1.0/hdf5-1.10.5-ltwm3rl6wginplgxx2p4ux44efi5qtek/bin

1) gcc/9.1.0-gcc-4.8.5-mj7s6dg   2) armadillo/9.900.3-gcc-9.1.0-qapqbay   3) atlas/3.10.2-gcc-9.1.0-jhgdopt   4) cmake/3.17.3-gcc-9.1.0-tsjr5x6   5) openmpi/4.1.1-gcc-9.1.0-fwmcvon   6) hdf5/1.10.5-gcc-9.1.0-ltwm3rl

#SBATCH --mail-type=<type>
#SBATCH --mail-user=<user>
https://slurm.schedmd.com/sbatch.html
https://wiki.vsc.ac.at/doku.php?id=doku:vsc4_queue

######LAST VERSION
1) gcc/9.1.0-gcc-4.8.5-mj7s6dg   2) armadillo/9.900.3-gcc-9.1.0-qapqbay   3) cmake/3.17.3-gcc-9.1.0-tsjr5x6   4) openmpi/4.1.1-gcc-9.1.0-fwmcvon   5) hdf5/1.10.5-gcc-9.1.0-ltwm3rl   6) miniconda3/4.10.3-gcc-9.1.0-nkcdatd
export PKG_CONFIG_PATH=$HOME/Software/3rd-Party/Install/lib/pkgconfig
conda activate /home/fs71764/miro/Software/3rd-Party/miniconda3/voxTrace