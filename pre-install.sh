#!/usr/bin/env bash
set -e

apt-get install autoconf automake autotools-dev build-essential cmake doxygen g++ gfortran gnuplot libarmadillo-dev libboost-dev libgsl-dev libhdf5-dev libtool meson ninja-build pkg-config python3 python3-dev python3-pip python3-setuptools python3-wheel sed wget
#cython cuda mlocate
pip3 install breathe cython numpy sphinx-rtd-theme

#apt-get nvidia-cuda-toolkit

#Install cuda 12.1

cd ..
mkdir installation
cd installation

wget http://lvserver.ugent.be/xraylib/xraylib-4.1.3.tar.gz && \
    tar -xzvf xraylib-4.1.3.tar.gz && \
    cd xraylib-4.1.3 && \
    ./configure && \
    make && \
    make install && \
    ldconfig && \
    cd .. && \
    rm -rf xraylib-4.1.3 xraylib-4.1.3.tar.gz

wget http://github.com/tschoonj/easyRNG/releases/download/easyRNG-1.2/easyRNG-1.2.tar.gz
  tar xfvz easyRNG-1.2.tar.gz --no-same-owner
  cd easyRNG-1.2
  mkdir build
  cd build
  meson ..
  ninja
  ninja test
  ninja install
  cd ../..
  
git clone https://github.com/PieterTack/polycap
    cd polycap
    git checkout 87b1c2a64d083d66919aff641c31c05028b851a5
    cd include
    sed -i '115a\POLYCAP_EXTERN' polycap-photon.h
    sed -i '116a\double polycap_scalar(polycap_vector3 vect1, polycap_vector3 vect2);' polycap-photon.h
    sed -i '117a\ ' polycap-photon.h
    sed -i '118a\POLYCAP_EXTERN' polycap-photon.h
    sed -i '119a\void polycap_norm(polycap_vector3 *vect);' polycap-photon.h
    sed -i '120a\ ' polycap-photon.h
    sed -i '121a\POLYCAP_EXTERN' polycap-photon.h
    sed -i '122a\polycap_leak* polycap_leak_new(polycap_vector3 leak_coords, polycap_vector3 leak_dir, polycap_vector3 leak_elecv, int64_t n_refl, size_t n_energies, double *weights, polycap_error **error);' polycap-photon.h
    sed -i '123a\ ' polycap-photon.h
    sed -i '124a\POLYCAP_EXTERN' polycap-photon.h
    sed -i '125a\int polycap_photon_within_pc_boundary(double polycap_radius, polycap_vector3 photon_coord, polycap_error **error);' polycap-photon.h
    sed -i '126a\ ' polycap-photon.h
    cd ..
    autoreconf -i
    ./configure
    make
    make install
    ldconfig
    cp config.h /usr/local/include/polycap/config.h
    cp src/polycap-private.h /usr/local/include/polycap/polycap-private.h
    cd ..


git clone https://github.com/PaNOSC-ViNYL/shadow3
    cd shadow3
    git checkout gfortran8-fixes
    cd src
    echo "Current working directory: $(pwd)"
    sed -i '9a\namespace Shadow3{' c/shadow_bind_cpp.hpp 
    sed -i '71i\}' c/shadow_bind_cpp.hpp 
    sed -i '3a\using namespace Shadow3;' c/shadow_bind_cpp.cpp
    sed -i '350s@^\s*.*@\tinstall ../shadow3 /usr/local/bin@' Makefile
    sed -i '351s@^\s*.*@\tinstall libshadow3.a libshadow3.so libshadow3c++.a libshadow3c.a libshadow3c.so /usr/local/lib@' Makefile
    sed -i '351a\\tmkdir -p /usr/local/include/shadow3' Makefile
    sed -i '352a\\tcp -r c/* /usr/local/include/shadow3/' Makefile
    sed -i '353a\\tcp -r def/* /usr/local/include/shadow3/' Makefile
    make 
    make lib
    make libstatic
    make install
    cd ../..

git clone https://github.com/sciplot/sciplot --recursive
    cd sciplot
    mkdir build && cd build
    cmake ..
    cmake --build . --target install
    cd ../..

wget https://ensmallen.org/files/ensmallen-2.19.1.tar.gz
    tar xfvz ensmallen-2.19.1.tar.gz --no-same-owner
    cd ensmallen-2.19.1 
    mkdir build
    cd build
    cmake ..
    make install
    cd ../..

 rm -r installation