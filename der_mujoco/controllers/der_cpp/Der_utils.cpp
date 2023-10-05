#include "Der_utils.h"
#include "Der_obj.h"
#include <string>
#include <iostream>
#include "Eigen/Dense"

const Eigen::Vector3d DerUtils::rotateVector3(const Eigen::Vector3d &v, const Eigen::Vector3d &u, const double a)
{
    Eigen::Matrix3d R;
    Eigen::Vector3d v_res;
    Eigen::Vector3d u_norm = u / u.norm();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (i == j) {
                R(i,j) = (
                    cos(a)
                    + (pow(u_norm(i),2.) * (1. - cos(a)))
                );
            }
            else {
                double ss = 1.;
                if (i < j) {ss *= -1.;}
                if (((i+1) * (j+1)) % 2 != 0) {ss *= -1.;}
                R(i,j) = (
                    u_norm(i) * u_norm(j) * (1. - cos(a))
                    + ss * u(3-(i+j)) * sin(a)
                );
            }
        }
    }
    v_res = R*v;
    return v_res;
}

const double DerUtils::calculateAngleBetween(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2)
{
    double cos_ab, ab;
    cos_ab = v1.dot(v2) / (v1.norm() * v2.norm());
    if (cos_ab > 1) {return 0;}
    ab = acos(cos_ab);
    if (std::isnan(ab)) {
        std::string str_error = "Error in DerUtils::calculateAngleBetween";
        std::cout << str_error << std::endl;
        std::cout << v1 << std::endl;
        std::cout << v2 << std::endl;
    }
    return ab;
}

const double DerUtils::calculateAngleBetween2(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2, const Eigen::Vector3d &v_anchor)
{
    double e_tol, sab, n_div, cab, ab;
    e_tol = 1e-3;
    sab = 0.;
    n_div = 0.;

    Eigen::Vector3d cross12;
    if ((v1-v2).norm() < e_tol) {return 0;}
    cross12 = v1.cross(v2);
    for (int i = 0; i < 3; i++) {
        if (std::abs(v_anchor(i)) < e_tol) {continue;}
        sab += (
            cross12(i)
            / (v1.norm() * v2.norm())
            / v_anchor(i)
        );
        n_div += 1.;
    }
    sab /= n_div;
    cab = (
        v1.dot(v2)
        / (v1.norm() * v2.norm())
    );
    ab = atan2(sab, cab);
    return ab;
}

const Eigen::Matrix3d DerUtils::createSkewSym(const Eigen::Vector3d &v)
{
    Eigen::Matrix3d skew_sym_v;
    skew_sym_v << 0., -v[2], v[1],
        v[2], 0., -v[0],
        -v[1], v[0], 0.;
    return skew_sym_v;
}

// int main()
// {
//     Eigen::Matrix3d bf1, bf2;
//     double a1, a2;
//     Eigen::Matrix3d ss1;

//     DerUtils derutils;
//     bf1.row(0) << 1., 0., 0.;
//     bf1.row(1) << 0., 1., 0.;
//     bf1.row(2) << 0., 0., 1.;
//     bf2 = bf1;
//     bf2.row(0) << 0., 0., 2.;
//     std::cout << bf2.row(0) << std::endl;
//     std::cout << bf1.row(0) << std::endl;

//     // bf2.x = derutils.rotateVector3(bf1.x, bf1.y, M_PI/4);
//     // std::cout << bf2.x << std::endl;

//     // Eigen::Vector3d v1, v2;
//     // v1 = bf1.x;
//     // v2 = bf1.y;
//     // double cos_ab, ab;
//     // cos_ab = v1.dot(v2) / (v1.norm() * v2.norm());
//     // if (cos_ab > 1) {return 0;}
//     // ab = acos(cos_ab);
//     // if (std::isnan(ab)) {
//     //     std::cout << v1 << std::endl;
//     //     std::cout << v2 << std::endl;
//     // }
//     // std::cout << ab << std::endl;

//     a1 = derutils.calculateAngleBetween(bf1.row(0), bf1.row(1));
//     std::cout << a1 << std::endl;

//     a2 = DerUtils::calculateAngleBetween2(bf1.row(0), bf1.row(1), -bf1.row(2));
//     std::cout << a2 << std::endl;
//     // a1 = bf2.x.dot(bf1.x);
//     // std::cout << a1 << std::endl;

//     // ss1 = DerUtils::createSkewSym(bf2.x);
//     // std::cout << ss1 << std::endl;
//     return 0;
// }
