#include "Der_iso.h"
#include "Der_obj.h"

// #include <iostream>
// #include <cmath>
// #include <vector>
// #include <chrono>
// #include <thread>
// #include <mutex>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Geometry"

/*
To-do:
- Make getthetan public

A_cpp(node_pos, bf0sim) --> (bfend)
A_py(bfend) --> (theta_n)
B_cpp(theta_n) --> (force_nodes)
*/

// std::mutex mtx;

DER_iso::DER_iso(
    int dim_np,
    double *node_pos,
    int dim_bf0,
    double *bf0sim,
    const double theta_n,
    const double overall_rot,
    const double a_bar,
    const double b_bar
)
{
    SegEdges e1;
    Vecnodes x1;

    // init variables
    d_vec = 0;
    nv = (int)(dim_np / 3) - 2 - d_vec * 2;
    bigL_bar = 0.;
    alpha_bar = a_bar;
    beta_bar = b_bar;
    for (int i = 0; i < (nv+1); i++) {
        edges.push_back(e1);
        nodes.push_back(x1);
    }
    // std::cout << edges.size() << std::endl;
    nodes.push_back(x1);

    j_rot << 0., -1., 1., 0.;
    
    edges[nv].theta = overall_rot;
    p_thetan = fmod(edges[nv].theta, (2. * M_PI));
    if (p_thetan > M_PI) {p_thetan -= 2 * M_PI;}
    // std::cin.get();

    initVars(dim_np, node_pos, dim_bf0, bf0sim);
    // for (int i = 0; i < (nv+2); i++) {
    //     std::cout << nodes[i].pos << std::endl;
    // }
}

void DER_iso::initVars(
    int dim_np,
    double *node_pos,
    int dim_bf0,
    double *bf0sim
)
{
    Eigen::Matrix3d init_mat3d;

    init_mat3d << 0., 0., 0.,
        0., 0., 0.,
        0., 0., 0.;
    for (int i = 1; i < (nv+1); i++) {
        for (int j = 0; j < 3; j ++) {
            nodes[i].nabkb.push_back(init_mat3d);
        }
    }

    bf0mat << bf0sim[0], bf0sim[1], bf0sim[2],
        bf0sim[3], bf0sim[4], bf0sim[5],
        bf0sim[6], bf0sim[7], bf0sim[8];

    update_XVecs(node_pos);
    updateX2E();
    updateE2K();
    updateE2Kb();
    // initBF(bf0sim);
    transfBF(bf0mat);
    // if abs(p_thetan - ptn) > 1e-6:
    //     raise Exception("Overall_rot and defined frame different")
    // initThreads();
}

// void DER_iso::initThreads()
// {
//     n_threads = 4;
//     s_i = 1;
//     e_i = nv + 1;
//     d_i = std::ceil((e_i - s_i) / n_threads);
//     over_i = (d_i*n_threads) - (e_i-s_i);

//     // std::cout << d_i << std::endl;

//     for (int n = 0; n < (n_threads-1); n++) {
//         nkbpsiThreads.push_back(
//             std::thread(&DER_iso::calculateNabKbandNabPsi_sub, this, (n*d_i + s_i), ((n+1)*d_i + s_i))
//         );
//     }
//     nkbpsiThreads.push_back(
//         std::thread(&DER_iso::calculateNabKbandNabPsi_sub, this, ((n_threads-1)*d_i + s_i), ((n_threads)*d_i + s_i - over_i))
//     );
//     for (std::thread &t : nkbpsiThreads) {
//         if (t.joinable()) {
//             t.join();
//         }
//     }
// }

bool DER_iso::updateVars(
    int dim_np,
    double *node_pos,
    int dim_bf0,
    double *bf0sim,
    int dim_bfe,
    double *bfesim
)
{
    bf0mat << bf0sim[0], bf0sim[1], bf0sim[2],
        bf0sim[3], bf0sim[4], bf0sim[5],
        bf0sim[6], bf0sim[7], bf0sim[8];

    bool bf_align;
    bf_align = true;
    
    update_XVecs(node_pos);
    updateX2E();
    updateE2K();
    updateE2Kb();
    // initBF(bf0sim);
    bf_align = transfBF(bf0mat);
    // updateTheta(theta_n);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            bfesim[3*i+j] = edges[nv].bf(i,j);
        }
    }
    return bf_align;
}

void DER_iso::update_XVecs(const double *node_pos)
{
    for (int i = 0; i < nv+2; i++) {
        nodes[i].pos << node_pos[3*i], node_pos[3*i+1], node_pos[3*i+2];
    }
}

void DER_iso::updateX2E()
{
    bigL_bar = 0;
    edges[0].e = nodes[1].pos - nodes[0].pos;
    edges[0].e_bar = edges[0].e.norm();
    for (int i = 1; i < nv+1; i++) {
        edges[i].e = nodes[i+1].pos - nodes[i].pos;
        edges[i].e_bar = edges[i].e.norm();
        edges[i].l_bar = edges[i].e_bar + edges[i-1].e_bar;
        bigL_bar += edges[i].l_bar;
    }
    bigL_bar /= 2.;
}

void DER_iso::updateE2K()
{
    for (int i = 1; i < nv+1; i++) {
        nodes[i].phi_i = DerUtils::calculateAngleBetween(edges[i-1].e, edges[i].e);
        nodes[i].k = 2. * tan(nodes[i].phi_i / 2.);
    }
}

void DER_iso::updateE2Kb()
{
    for (int i = 1; i < nv+1; i++) {
        nodes[i].kb = (
            2. * edges[i-1].e.cross(edges[i].e)
            / (
                edges[i-1].e_bar * edges[i].e_bar
                + edges[i-1].e.dot(edges[i].e)
            )
        );
    }
}

// void DER_iso::initBF(Eigen::Matrix3d &bf_0)
// {
    // double parll_tol = 1e-6;
    // Eigen::Vector3d wa1;
    // wa1 = {0., 0., 1.};
    // bf0_bar.row(0) = edges[0].e / edges[0].e_bar;
    // bf0_bar.row(1) = bf0_bar.row(0).cross(wa1);
    // if (bf0_bar.row(1).norm() < parll_tol) 
    // {
    //     wa1 = {0., 1., 0.};
    //     bf0_bar.row(1) = bf0_bar.row(0).cross(wa1);
    // }
    // bf0_bar.row(1) /= bf0_bar.row(1).norm();
    // bf0_bar.row(2) = bf0_bar.row(0).cross(bf0_bar.row(1));
    // edges[0].bf = bf0_bar;
// }

bool DER_iso::transfBF(const Eigen::Matrix3d &bf_0)
{
    bool bf_align;
    bf_align = true;

    edges[0].bf = bf_0;
    
    for (int i = 1; i < nv+1; i++) {
        edges[i].bf.row(0) = edges[i].e / edges[i].e.norm();
        if (nodes[i].kb.norm() == 0) {edges[i].bf.row(1) = edges[i-1].bf.row(1);}
        else {
            edges[i].bf.row(1) = DerUtils::rotateVector3(
                edges[i-1].bf.row(1),
                nodes[i].kb / nodes[i].kb.norm(),
                nodes[i].phi_i
            );
            if (std::abs(edges[i].bf.row(1).dot(edges[i].bf.row(0))) > 1e-1) {
                bf_align = false;
                // throw std::invalid_argument("Bishop transfer error: axis 1 not perpendicular to axis 0.");
            }
        }
        edges[i].bf.row(2) = edges[i].bf.row(0).cross(edges[i].bf.row(1));
    }
    return bf_align;
}

void DER_iso::updateThetaN(const double theta_n)
{
    double diff_theta = theta_n - p_thetan;
    int t_n_whole;

    // acct for 2pi rotation
    if (abs(diff_theta) < (M_PI / 4)) {
        edges[nv].theta += diff_theta;
    }
    else if (diff_theta > 0.) {
        edges[nv].theta += diff_theta - (2 * M_PI);
    }
    else {
        edges[nv].theta += diff_theta + (2 * M_PI);
    }
    // correction step: match self.theta[-1] to theta_n
    t_n_whole = (int) (edges[nv].theta / (2 * M_PI));
    // std::cout << t_n_whole << std::endl;
    // accounting for int rounding problem
    // (+ve rounds down, -ve rounds up)
    if ((edges[nv].theta < 0.) && (theta_n > 0.)) {
        t_n_whole -= 1;
    }
    if ((edges[nv].theta > 0.) && (theta_n < 0.)) {
        t_n_whole += 1;
    }

    edges[nv].theta = t_n_whole * (2 * M_PI) + theta_n;
    p_thetan = theta_n;
}

double DER_iso::updateTheta(const double theta_n)
{
    double d_theta;

    updateThetaN(theta_n);
    
    // if (std::isnan(edges[nv].theta)) {std::cout << edges[nv].theta << std::endl;}
    // if (std::isnan(edges[nv].theta)) {std::cout << 't' << std::endl;}
    
    d_theta = (edges[nv].theta - edges[0].theta) / nv;
    for (int i = 0; i < (nv+1); i++) {
        edges[i].theta = d_theta * i;
    }
    // std::cout << 'q' << std::endl;
    return edges[nv].theta;
}

void DER_iso::resetTheta(const double theta_n, const double overall_rot)
{
    p_thetan = theta_n;
    edges[nv].theta = overall_rot;
}

void DER_iso::changeAlphaBeta(const double a_bar, const double b_bar)
{
    alpha_bar = a_bar;
    beta_bar = b_bar;
}

// void DER_iso::calculateNabKbandNabPsi()
// {
//     int n_change;
//     n_change = -1;
//     for (int n = 0; n < (n_threads-1); n++) {
//         nkbpsiThreads[n] = (
//             std::thread(&DER_iso::calculateNabKbandNabPsi_sub2, this, (n*d_i + s_i), ((n+1)*d_i + s_i))
//         );
//     }
//     if (((n_threads)*d_i + s_i - over_i) - ((n_threads-1)*d_i + s_i) > 0) {
//         nkbpsiThreads[n_threads-1] = (
//             std::thread(&DER_iso::calculateNabKbandNabPsi_sub2, this, ((n_threads-1)*d_i + s_i), ((n_threads)*d_i + s_i - over_i))
//         );
//         n_change = 0;
//     }
//     for (std::thread &t : nkbpsiThreads) {
//         if (t.joinable()) {
//             t.join();
//         }
//     }

//     int split_id;
//     if (d_i > 1) {
//         for (int n = 1; n < (n_threads + n_change); n++) {
//             split_id = (n*d_i + s_i);
//             nodes[split_id-1].force += nodes[split_id-1].force_sub;
//             nodes[split_id].force += nodes[split_id].force_sub;
//             // std::cout << split_id << 'R' << std::endl;
//         }
//     }
//     else {
//         split_id = (d_i + s_i - 1);
//         nodes[split_id].force += nodes[split_id].force_sub;
//         for (int n = 1; n < (n_threads + n_change); n++) {
//             split_id = (n*d_i + s_i);
//             nodes[split_id].force += nodes[split_id].force_sub;
//             // std::cout << split_id << 'R' << std::endl;
//         }
//     }
// }

// void DER_iso::calculateNabKbandNabPsi_sub(const int start_i, const int end_i)
// {
//     for (int i = start_i; i < end_i; i++) {
//         nodes[i].nabkb[0] = (
//             2 * DerUtils::createSkewSym(edges[i].e)
//             + (nodes[i].kb * edges[i].e.transpose())
//             / (
//                 edges[i-1].e_bar * (edges[i].e_bar)
//                 + edges[i-1].e.dot(edges[i].e)
//             )
//         );
//         nodes[i].nabkb[2] = (
//             2 * DerUtils::createSkewSym(edges[i-1].e)
//             - (nodes[i].kb * edges[i-1].e.transpose())
//             / (
//                 edges[i-1].e_bar * (edges[i].e_bar)
//                 + edges[i-1].e.dot(edges[i].e)
//             )
//         );
//         nodes[i].nabkb[1] = (- (nodes[i].nabkb[0] + nodes[i].nabkb[2]));
//         nodes[i].nabpsi.row(0) = nodes[i].kb / (2 * edges[i-1].e_bar);
//         nodes[i].nabpsi.row(2) = - nodes[i].kb / (2 * edges[i].e_bar);
//         nodes[i].nabpsi.row(1) = - (nodes[i].nabpsi.row(0) + nodes[i].nabpsi.row(2));
//     }  
// }

void DER_iso::calculateNabKbandNabPsi_sub2(const int start_i, const int end_i)
{
    for (int i = start_i; i < end_i; i++) {
        nodes[i].nabkb[0] = (
            (
                2 * DerUtils::createSkewSym(edges[i].e)
                + (nodes[i].kb * edges[i].e.transpose())
            )
            / (
                edges[i-1].e_bar * (edges[i].e_bar)
                + edges[i-1].e.dot(edges[i].e)
            )
        );
        nodes[i].nabkb[2] = (
            (
                2 * DerUtils::createSkewSym(edges[i-1].e)
                - (nodes[i].kb * edges[i-1].e.transpose())
            )
            / (
                edges[i-1].e_bar * (edges[i].e_bar)
                + edges[i-1].e.dot(edges[i].e)
            )
        );
        nodes[i].nabkb[1] = (- (nodes[i].nabkb[0] + nodes[i].nabkb[2]));
        nodes[i].nabpsi.row(0) = nodes[i].kb / (2 * edges[i-1].e_bar);
        nodes[i].nabpsi.row(2) = - nodes[i].kb / (2 * edges[i].e_bar);
        nodes[i].nabpsi.row(1) = - (nodes[i].nabpsi.row(0) + nodes[i].nabpsi.row(2));
        
        
        // nodes[i].force << 0., 0., 0.;
        for (int j = (i-1); j < (i+2); j++) {
            if ((j > nv) || (j<1)) {continue;}
            // if ((j < start_i) || (j > end_i)) {mtx.lock();}
            // if ((j < start_i) || (j > (end_i-1))) {
            //     // mtx.lock();
            //     nodes[j].force_sub += - (
            //         2. * alpha_bar
            //         * nodes[i].nabkb[j-i+1].transpose() * nodes[i].kb
            //     ) / edges[i].l_bar;
            //     nodes[j].force_sub += (
            //         beta_bar * (edges[nv].theta - edges[0].theta)
            //         * nodes[i].nabpsi.row(j-i+1)
            //     ) / bigL_bar;

            //     // std::cout << start_i << 'S' << end_i << 'E' << j << 'A' << std::endl;
            //     // mtx.unlock();
            //     continue;
            // }
            nodes[j].force += - (
                2. * alpha_bar
                * nodes[i].nabkb[j-i+1].transpose() * nodes[i].kb
            ) / edges[i].l_bar;
            nodes[j].force += (
                beta_bar * (edges[nv].theta - edges[0].theta)
                * nodes[i].nabpsi.row(j-i+1)
            ) / bigL_bar;
            // Eigen::Vector3d f2oo;
            // double f2oo;
            // f2oo = edges[nv].theta;
            // if (std::isnan(nodes[j].force[0])) {std::cout << nodes[j].force[0] << std::endl;}
            // if (std::isnan(nodes[j].force[1])) {std::cout << edges[nv].theta << std::endl;}
            // if (std::isnan(nodes[j].force[2])) {std::cout << 'c' << std::endl;}
            // if ((j < start_i) || (j > end_i)) {mtx.unlock();}
        }
    }
    
    //     for (int j = (i-1); j < (i+1); j++) {
    //         // std::cout << (nodes[j].nabkb.size()) << std::endl;
    //         if ((j > nv) || (j<start_i)) {continue;}
    //         nodes[i].force += - (
    //             2. * alpha_bar
    //             * nodes[j].nabkb[i-j+1].transpose() * nodes[j].kb
    //         ) / edges[j].l_bar;
    //         // std::cout << i << std::endl;
    //         // std::cout << j << std::endl;
    //         // std::cout << nodes[j].nabkb[i-j+1].transpose() * nodes[i].kb << std::endl;
    //         // std::cout << 1111 << std::endl;
    //         // std::cout << nodes[i].nabkb[i-j+1].transpose() << std::endl;
    //         // std::cout << nodes[i].kb << std::endl;
    //         // std::cout << nodes[i].force << std::endl;
    //         nodes[i].force += (
    //             beta_bar * (edges[nv].theta - edges[0].theta)
    //             * nodes[j].nabpsi.row(i-j+1)
    //         ) / bigL_bar;
    //         // std::cout << nodes[i].force << std::endl;
    //         // std::cin.get();
    //     }
    //     if ((i-1) < 1) {continue;}
    //     nodes[i-1].force += - (
    //         2. * alpha_bar
    //         * nodes[i].nabkb[0].transpose() * nodes[i].kb
    //     ) / edges[i].l_bar;
        
    //     nodes[i-1].force += (
    //         beta_bar * (edges[nv].theta - edges[0].theta)
    //         * nodes[i].nabpsi.row(0)
    //     ) / bigL_bar;
    // }
    // nodes[end_i].force += - (
    //     2. * alpha_bar
    //     * nodes[end_i-1].nabkb[2].transpose() * nodes[end_i-1].kb
    // ) / edges[end_i-1].l_bar;
    
    // nodes[end_i].force += (
    //     beta_bar * (edges[nv].theta - edges[0].theta)
    //     * nodes[end_i-1].nabpsi.row(2)
    // ) / bigL_bar;
}

// void DER_iso::calculateCenterlineF()
// {
//     using namespace std::literals::chrono_literals;
//     auto start = std::chrono::high_resolution_clock::now();
//     if (false) {
//         calculateNabKbandNabPsi();
//     }
//     else {
//         calculateNabKbandNabPsi_sub(1, nv+1);
//     }
//     auto stop1 = std::chrono::high_resolution_clock::now();
//     for (int i = 1; i < (nv+1); i++) {
//         nodes[i].force << 0., 0., 0.;
//         for (int j = (i-1); j < (i+2); j++) {
//             // std::cout << (nodes[j].nabkb.size()) << std::endl;
//             if ((j > nv) || (j<1)) {continue;}
//             nodes[i].force += - (
//                 2. * alpha_bar
//                 * nodes[j].nabkb[i-j+1].transpose() * nodes[j].kb
//             ) / edges[j].l_bar;
//             // std::cout << i << std::endl;
//             // std::cout << j << std::endl;
//             // std::cout << nodes[j].nabkb[i-j+1].transpose() * nodes[i].kb << std::endl;
//             // std::cout << 1111 << std::endl;
//             // std::cout << nodes[i].nabkb[i-j+1].transpose() << std::endl;
//             // std::cout << nodes[i].kb << std::endl;
//             // std::cout << nodes[i].force << std::endl;
//             nodes[i].force += (
//                 beta_bar * (edges[nv].theta - edges[0].theta)
//                 * nodes[j].nabpsi.row(i-j+1)
//             ) / bigL_bar;
//             // std::cout << nodes[i].force << std::endl;
//             // std::cin.get();
//         }
//     }
//     auto stop2 = std::chrono::high_resolution_clock::now();

//     std::chrono::duration<float> duration1, duration2;
//     duration1 = stop1 - start;
//     duration2 = stop2 - stop1;
//     // std::cout << duration1.count() << "A" << std::endl;
//     // std::cout << duration2.count() << "B" << std::endl;
// }

void DER_iso::calculateCenterlineF2(int dim_nf, double *node_force)
{
    for (int i = 1; i < (nv+1); i++) {
        nodes[i].force << 0., 0., 0.;
        // nodes[i].force_sub << 0., 0., 0.;
    }

    calculateNabKbandNabPsi_sub2(1, nv+1);

    for (int i = 1; i < (nv+1); i++) {
        for (int j = 0; j < 3; j++) {
            node_force[3*i+j] = nodes[i].force(j);
        }
    }
    
    // using namespace std::literals::chrono_literals;
    // auto start = std::chrono::high_resolution_clock::now();
    // if (false) {
    //     calculateNabKbandNabPsi();
    // }
    // else {
    //     calculateNabKbandNabPsi_sub2(1, nv+1);
    // }
    // auto stop1 = std::chrono::high_resolution_clock::now();
    // auto stop2 = std::chrono::high_resolution_clock::now();

    // std::chrono::duration<float> duration1, duration2;
    // duration1 = stop1 - start;
    // duration2 = stop2 - stop1;
    // std::cout << duration1.count() << "A" << std::endl;
    // std::cout << duration2.count() << "B" << std::endl;
}

// misc calcs
void DER_iso::initQe_o2m_loc(
    int dim_qo2m,
    double *q_o2m
    // int dim_qm2o,
    // double *q_m2o,
)
{
    qe_o2m_loc.x() = q_o2m[0];
    qe_o2m_loc.y() = q_o2m[1];
    qe_o2m_loc.z() = q_o2m[2];
    qe_o2m_loc.w() = q_o2m[3];

    qe_o2m_loc = qe_o2m_loc.normalized();

    // qe_m2o_loc.x() = q_m2o[0];
    // qe_m2o_loc.y() = q_m2o[1];
    // qe_m2o_loc.z() = q_m2o[2];
    // qe_m2o_loc.w() = q_m2o[3];
}

void DER_iso::calculateOf2Mf(
    int dim_mato, double *mat_o,
    int dim_matres, double *mat_res
)
{
    Eigen::Matrix3d mat_mato;
    Eigen::Matrix3d mat_result;
    mat_mato << mat_o[0], mat_o[1], mat_o[2],
        mat_o[3], mat_o[4], mat_o[5],
        mat_o[6], mat_o[7], mat_o[8];

    Eigen::Quaterniond q_mato(mat_mato);
    // Eigen::Quaterniond q_mid;
    // q_mid = q_mato * qe_o2m_loc
    mat_result = (q_mato * qe_o2m_loc).normalized().toRotationMatrix();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            mat_res[j+3*i] = mat_result(i,j);
        }
    }
}

double DER_iso::angBtwn3(
    int dim_v1, double *v1,
    int dim_v2, double *v2,
    int dim_va, double *va
)
{
    Eigen::Vector3d v1_c(v1[0], v1[1], v1[2]);
    Eigen::Vector3d v2_c(v2[0], v2[1], v2[2]);
    Eigen::Vector3d va_c(va[0], va[1], va[2]);
    double theta_diff, dot_norm_val;
    // v1_c << v1[0], v1[1], v1[2];
    // v2_c << v2[0], v2[1], v2[2];
    // va_c << va[0], va[1], va[2];
    dot_norm_val = v1_c.dot(v2_c)/(v1_c.norm()*v2_c.norm());
    if (dot_norm_val > 1.) {dot_norm_val = 1.;}
    theta_diff = acos(dot_norm_val);
    // std::cout << v1_c << std::endl;
    // std::cout << v2_c << std::endl;
    // std::cout << v1_c.dot(v2_c) << std::endl;
    // std::cout << (v1_c.norm()*v2_c.norm()) << std::endl;
    // std::cout << (v1_c.dot(v2_c)/(v1_c.norm()*v2_c.norm())) << std::endl;
    // std::cout << acos(v1_c.dot(v2_c)/(v1_c.norm()*v2_c.norm())) << std::endl;
    // std::cout << (dot_norm_val > 1.) << std::endl;
    // std::cout << (dot_norm_val == 1) << std::endl;
    
    // std::cout << theta_diff << std::endl;
    // std::cout << 'z' << std::endl;
    if ((v1_c.cross(v2_c)).dot(va_c) < 0) {theta_diff *= -1.;}
    return theta_diff;
}