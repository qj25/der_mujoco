"""
To-do:
"""

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

import gymnasium
import random
from time import time
import der_mujoco.utils.transform_utils as T

from der_mujoco.utils.transform_utils import IDENTITY_QUATERNION
from der_mujoco.envs import DerRope1DEnv

def lhb_data(
    alpha_bar,
    beta_bar,
    s,
    m,
    t,
    e_x,
):
    # for analytical results, tanh^2(s_ss)
    phi = np.arccos(np.dot(t,e_x))
    phi_0 = max(phi)
    dal = (
        beta_bar * m / (2. * alpha_bar)
        * np.sqrt((1 - np.cos(phi_0))/(1 + np.cos(phi_0)))
    ) * s
    
    func_phi = (
        (np.cos(phi) - np.cos(phi_0))
        / (1 - np.cos(phi_0))
    )
    
    return dal, func_phi

def mbi_data(
    beta_bar,
    theta_crit,
    alpha_bar=1.,
):
    b_a = beta_bar / alpha_bar
    return b_a, theta_crit

def mbi_plot(b_a, theta_crit):
    b_a_base = b_a.copy()
    theta_crit_base = 2*np.pi*np.sqrt(3)/(b_a_base)

    max_devi_theta_crit = np.max(np.abs(theta_crit_base - theta_crit))
    print(f"max_devi_theta_crit = {max_devi_theta_crit}")

    plt.figure("Michell's Buckling Instability")
    plt.xlabel(r"$\beta/\alpha$")
    plt.ylabel(r'$\theta^n$')
    plt.plot(b_a_base, theta_crit_base)
    plt.plot(b_a, theta_crit)
    plt.legend(['Analytical','Simulation'])
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def mbi_indivtest(
    overall_rot=0.,
    alpha_val=1.,
    beta_val=1.,
    do_render=False,
    new_start=False,
):
    r_pieces = 51
    r_len = 2*np.pi * r_pieces / (r_pieces-1)
    r_mass = r_len * 2.
    env = DerRope1DEnv(
        overall_rot=overall_rot,
        do_render=do_render,
        r_pieces=r_pieces,
        r_len=r_len,
        test_type='mbi',
        alpha_bar=alpha_val,
        beta_bar=beta_val,
        r_mass=r_mass,
        new_start=new_start
    )
    if not env.circle_oop:
        theta_crit = 0.
    else:
        theta_crit = overall_rot
    
    # env.do_render = False
    env.close()
    return theta_crit

def mbi_test(new_start=False, load_from_pickle=False, do_render=False):
    mbi_picklename = 'mbi1.pickle'
    mbi_picklename = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "der_mujoco/data/" + mbi_picklename
    )
    if load_from_pickle:
        print('Loading MBI test...')
        with open(mbi_picklename, 'rb') as f:
            pickle_mbidata = pickle.load(f)
        idhalf_pickle = round(len(pickle_mbidata)/2)
        b_a = pickle_mbidata[:idhalf_pickle]
        theta_crit = pickle_mbidata[idhalf_pickle:]
    else:
        print('Starting MBI test...')
        n_data_mbi = 11
        beta_bar_lim = np.array([0.5, 1.25])
        beta_bar_step = (beta_bar_lim[1] - beta_bar_lim[0]) / (n_data_mbi - 1)
        beta_bar = np.zeros(n_data_mbi)
        alpha_bar = np.zeros(n_data_mbi)
        for i in range(n_data_mbi):
            beta_bar[i] = beta_bar_step * i + beta_bar_lim[0]
            alpha_bar[i] = 1.
        beta_bar = beta_bar[::-1]
        b_a = beta_bar / alpha_bar
        theta_crit = np.zeros(n_data_mbi)
        if new_start:
            theta_crit[i] = mbi_indivtest(
                alpha_val=alpha_bar[i],
                beta_val=beta_bar[i],
                new_start=new_start,
                do_render=do_render
            )
        overall_rot = 8.5
        for i in range(n_data_mbi):
            print(f'b_a = {b_a[i]}')
            while theta_crit[i] < 1e-7:
                print(f'overall_rot = {overall_rot}')
                theta_crit[i] = mbi_indivtest(
                    overall_rot=overall_rot,
                    alpha_val=alpha_bar[i],
                    beta_val=beta_bar[i],
                    do_render=do_render
                )
                overall_rot += 0.1 # * (np.pi/180)
        pickle_mbidata = np.concatenate((b_a,theta_crit))
        with open(mbi_picklename, 'wb') as f:
            pickle.dump(pickle_mbidata,f)
            print('mbi test data saved!')
    
    mbi_plot(b_a=b_a, theta_crit=theta_crit)

def _pickle2data(ax, r_pieces):
    pickledata_path = 'lhb{}.pickle'.format(r_pieces)
    pickledata_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "der_mujoco/data/" + pickledata_path
    )
    # input(pickledata_path)
    with open(pickledata_path, 'rb') as f:
        pickledata = pickle.load(f)
    s_ss_base2 = pickledata[1]/12*18.26
    fphi_base2 = (np.tanh(s_ss_base2))**2.
    id_counter = 0
    # print(s_ss_base2)
    while id_counter < len(s_ss_base2):
        # print(id_counter)
        # print(s_ss_base2[id_counter])
        # print(s_ss_base2[id_counter] <= 6)
        # print(s_ss_base2[id_counter] > 6)
        if s_ss_base2[id_counter] <= -6 or s_ss_base2[id_counter] > 6:
            s_ss_base2 = np.delete(s_ss_base2, id_counter, axis=0)
            fphi_base2 = np.delete(fphi_base2, id_counter, axis=0)
            pickledata[0] = np.delete(pickledata[0], id_counter, axis=0)
            # print(s_ss_base2)
        else:
            id_counter += 1
    avg_devi_lhb = np.linalg.norm(pickledata[0] - fphi_base2) / r_pieces
    print(f"Average deviation for {r_pieces} pieces = {avg_devi_lhb}")
    s_ss_base2 = np.insert(s_ss_base2, 0, -6.)
    fphi_base2 = np.insert(fphi_base2, 0, 1.)
    pickledata[0] = np.insert(pickledata[0], 0, 1.)
    s_ss_base2 = np.insert(s_ss_base2, len(s_ss_base2), 6.)
    fphi_base2 = np.insert(fphi_base2, len(fphi_base2), 1.)
    pickledata[0] = np.insert(pickledata[0], len(pickledata[0]), 1.)
    ax.plot(s_ss_base2, pickledata[0], alpha=0.5)
    # print(s_ss_base2)
    # input()
    

def lhb_plot(r_pieces_list):
    s_ss_base = np.arange(-6., 6., 0.01)
    fphi_base = (np.tanh(s_ss_base))**2.
    fig = plt.figure("Localized Helical Buckling", figsize=(6,4))
    ax = fig.add_subplot(111)
    ax.set_xlabel("s/s*")
    ax.set_ylabel(r'$f(\varphi)$')
    ax.plot(s_ss_base, fphi_base, linewidth="2", alpha=0.5)
    
    ax.spines['left'].set_position(('data',0))
    ax.spines['bottom'].set_position(('data',0))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.labelpad = 0.

    legend_str = []
    legend_str.append('Analytical')
    for r_pieces in r_pieces_list:
        _pickle2data(ax, r_pieces=r_pieces)
        legend_str.append(str(r_pieces-1))

    ax.legend(legend_str)
    plt.tight_layout()
    # plt.savefig('test1.svg')
    plt.show()

def lhb_indivtest(
    r_pieces=20,
    do_render=False,
    new_start=False
):
    alpha_val = 1.345
    beta_val = 0.789
    overall_rot = 27. * (2*np.pi)
    r_len = 9.29
    r_mass = r_len * 2.
    env = DerRope1DEnv(
        do_render=do_render,
        r_pieces=r_pieces,
        r_len=r_len,
        test_type='lhb',
        overall_rot=overall_rot,
        alpha_bar=alpha_val,
        beta_bar=beta_val,
        r_mass=r_mass,
        new_start=new_start,
        limit_f=True
    )
    # env.do_render = False
    # env.reset()
    env.close()

def lhb_test(new_start=True, load_from_pickle=False, do_render=False):
    print('Starting LHB test...')
    # n_pieces = [60]
    n_pieces = [41, 61, 81, 111, 141, 181]
    # n_pieces = [111, 141]
    if not load_from_pickle:
        for i in n_pieces:
            lhb_indivtest(
                r_pieces=i,
                new_start=new_start,
                do_render=do_render    
            )
    lhb_plot(r_pieces_list=n_pieces)

# lhb_test(new_start=False, load_from_pickle=True, do_render=False)
mbi_test(new_start=False, load_from_pickle=False, do_render=False)
