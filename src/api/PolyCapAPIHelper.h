/*
 * Copyright (C) 2018 Pieter Tack, Tom Schoonjans and Laszlo Vincze
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * */
#include <polycap.h>

struct _polycap_profile
  {
  int nmax;
  double *z;
  double *cap;
  double *ext;
  };

struct _polycap_description
  {
  double sig_rough;
  int64_t n_cap;
  double open_area;
  unsigned int nelem;
  int *iz;
  double *wi;
  double density;
  polycap_profile *profile;
  };

struct _polycap_source
  {
  polycap_description *description;
  polycap_rng *rng;
  double d_source;
  double src_x;
  double src_y;
  double src_sigx;
  double src_sigy;
  double src_shiftx;
  double src_shifty;
  double hor_pol;
  size_t n_energies;
  double *energies;
  };

struct _polycap_photon
  {
  polycap_description *description;
  polycap_leak **extleak;
  polycap_leak **intleak;
  int64_t n_extleak;
  int64_t n_intleak;
  polycap_vector3 start_coords;
  polycap_vector3 start_direction;
  polycap_vector3 start_electric_vector;
  polycap_vector3 exit_coords;
  polycap_vector3 exit_direction;
  polycap_vector3 exit_electric_vector;
  polycap_vector3 src_start_coords;
  size_t n_energies;
  double *energies;
  double *weight;
  double *amu;
  double *scatf;
  int64_t i_refl;
  double d_travel;
  };

struct _polycap_transmission_efficiencies
  {
  size_t n_energies;
  double *energies;
  double *efficiencies;
  struct _polycap_images *images;
  polycap_source *source;
  };

struct _polycap_images
  {
  int64_t i_start;
  int64_t i_exit;
  double *src_start_coords[2];
  double *pc_start_coords[2];
  double *pc_start_dir[2];
  double *pc_start_elecv[2];
  double *pc_exit_coords[3];
  double *pc_exit_dir[2];
  double *pc_exit_elecv[2];
  int64_t *pc_exit_nrefl;
  double *pc_exit_dtravel;
  double *exit_coord_weights;
  int64_t i_extleak;
  double *extleak_coords[3];
  double *extleak_dir[2];
  double *extleak_coord_weights;
  int64_t *extleak_n_refl;
  int64_t i_intleak;
  double *intleak_coords[3];
  double *intleak_dir[2];
  double *intleak_elecv[2];
  double *intleak_coord_weights;
  int64_t *intleak_n_refl;
  };



