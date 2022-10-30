#ifndef DERISO_H
#define DERISO_H

#include "Der_obj.h"
#include "Der_utils.h"
#include <Eigen/Core>
// #include <thread>

class DER_iso
{
public:
    DER_iso(
        int dim_np,
        double *node_pos,
        int dim_bf0,
        double *bf0sim,
        const double theta_n,
        const double overall_rot,
        const double a_bar,
        const double b_bar
    );

    // ~DER_iso();
    
    std::vector <Vecnodes, Eigen::aligned_allocator<Vecnodes> > nodes;
    double overall_rot;

    int d_vec;
    int nv;
    std::vector <SegEdges, Eigen::aligned_allocator<SegEdges> > edges;
    
    // init _variables
    double bigL_bar;
    Eigen::Matrix3d bf0_bar;

    // define variable constants
    double alpha_bar;
    double beta_bar;
    Eigen::Matrix2d j_rot;
    
    double p_thetan;

    // Cpp2Py vars
    Eigen::Matrix3d bf0mat;

    // // threading variables
    // int s_i, e_i, d_i, n_threads, over_i;
    // std::vector<std::thread> nkbpsiThreads;
    // // std::vector <Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > force_sub;
    
    // Functions:
    // void initVars(
    //     const std::vector <Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &node_pos, 
    //     const Eigen::Matrix3d &bf0sim
    // );  //
    bool updateVars(
        int dim_np,
        double *node_pos,
        int dim_bf0,
        double *bf0sim,
        int dim_bfe,
        double *bfesim
    );  //

    void calculateCenterlineF2(int dim_nf, double *node_force);  // multithreading w/ force-nab loop share

    double updateTheta(double theta_n);   //

    void resetTheta(double theta_n, double overall_rot); //

private:
    void initVars(
        int dim_np,
        double *node_pos,
        int dim_bf0,
        double *bf0sim
    );  //
    // void initThreads(); //

    // void updateVars(
    //     const std::vector <Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &node_pos,
    //     const Eigen::Matrix3d &bf0sim
    // );  //

    // void update_XVecs(const std::vector <Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &node_pos);
    void update_XVecs(const double *node_pos);
    
    void updateX2E();   //
    void updateE2K();   //
    void updateE2Kb();  //
    // void initBF(Eigen::Matrix3d &bf_0);  //
    bool transfBF(const Eigen::Matrix3d &bf_0);    //
    
    void updateThetaN(double theta_n);  //

    // main calculation
    // void calculateNabKbandNabPsi(); // multithreading
    // void calculateNabKbandNabPsi_sub(const int start_i, const int end_i); //
    void calculateNabKbandNabPsi_sub2(const int start_i, const int end_i); // force-nab loop share

    // void calculateCenterlineF();  // multithreading

    // // bring this to python (maybe)
    // void dampForce();
    // void limitForce();
    // void limitTotalForce();

    // void updateForce(
    //     const std::vector <Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > &node_pos,
    //     double overall_rot,
    //     double theta_n
    // );

    // double calculateEBend();
    // double calculateETwist();

    // Function to help integration w/ Python
    
};

#endif