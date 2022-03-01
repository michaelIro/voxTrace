#!/bin/bash
module load gcc/9.1.0-gcc-4.8.5-mj7s6dg armadillo/9.900.3-gcc-9.1.0-mht5noj cmake/3.17.3-gcc-9.1.0-tsjr5x6 openmpi/4.1.1-gcc-9.1.0-fwmcvon hdf5/1.12.0-gcc-9.1.0-x63yn53 miniconda3/4.10.3-gcc-9.1.0-nkcdatd gnuplot/5.2.8-gcc-9.1.0-2dqwfce boost/1.70.0-gcc-9.1.0-5bbbzzb
export PKG_CONFIG_PATH=$HOME/Software/3rd-Party/Install/lib/pkgconfig
conda activate /home/fs71764/miro/Software/3rd-Party/miniconda3/voxTrace