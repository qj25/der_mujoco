%module Der_iso
%{
#define SWIG_FILE_WITH_INIT
// #include <iostream>
#include "Der_obj.h"
#include "Der_utils.h"
#include "Der_iso.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

%apply (int DIM1, double* IN_ARRAY1) {
    (int dim_np, double* node_pos),
    (int dim_bf0, double* bf0sim),
    (int dim_bfe, double* bfesim),
    (int dim_nf, double* node_force),
    (int dim_qo2m, double* q_o2m),
    // (int dim_qm2o, double* q_m2o),
    (int dim_mato, double* mat_o),
    (int dim_matres, double* mat_res),
    (int dim_v1, double *v1),
    (int dim_v2, double *v2),
    (int dim_va, double *va)
};
// %apply (int DIM1, double* ARGOUT_ARRAY1) {(int dim_nf, double* node_force)};


%include "Der_iso.h"
// %include "Der_obj.h"
// %include "Der_utils.h"


// class DER_iso
// {
// public:
//     DER_iso(
//         int dim_np,
//         double *node_pos,
//         int dim_bf0,
//         double *bf0sim,
//         const double theta_n,
//         const double overall_rot
//     );

//     ~DER_iso();

//     void updateVars(
//         int dim_np,
//         double *node_pos,
//         int dim_bf0,
//         double *bf0sim
//     );  //

//     void calculateCenterlineF2(int dim_nf, double *node_force);
// };

// %include "Der_obj.h"
// %include "Der_utils.h"


// #include <Eigen/Dense>
// #include <Eigen/Core>

// extern DER_iso(
//     int dim_np,
//     double *node_pos,
//     int dim_bf0,
//     double *bf0sim,
//     const double theta_n,
//     const double overall_rot
// );

// extern DER_iso::updateVars(
//         int dim_np,
//         double *node_pos,
//         int dim_bf0,
//         double *bf0sim
//     );  //

// extern DER_iso::calculateCenterlineF2(int dim_nf, double *node_force);

/*
    python3-config --cflags
    swig -c++ -python -o Der_iso_wrap.cpp Der_iso.i

    g++ -c -fpic Der_iso.cpp Der_utils.cpp -std=c++14
    g++ -c -fpic Der_iso_wrap.cpp -I/home/qj/anaconda3/envs/rlenv/include/python3.6m -std=c++14
    g++ -shared Der_iso.o Der_utils.o Der_iso_wrap.o _Der_iso.so

    g++ -c Der_iso.cpp Der_iso_wrap.cpp -I/home/qj/anaconda3/envs/rlenv/include/python3.6m -fPIC -std=c++14
    ld -shared Der_iso.o Der_iso_wrap.o -o _Der_iso.so -fPIC

    g++ -Wl,--gc-sections -fPIC -shared -lstdc++ Der_iso.o Der_iso_wrap.o -o _Der_iso.so
*/