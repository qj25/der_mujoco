"""
Discrete Elastic Rope controller on a no-damping, no-spring
finite element rod.

Assumptions:
- one end is fixed (xml id 1 if one end, xml id -1 if both ends)
- equality constraint of endpoint stable

Drawbacks:
- tuning of force limits (f_lim) required

Possible solutions:
- raise limit (remove is best)
- adjust mass
- change vel_reset freq (5-10 --> requires more testing)
- adjust connect constraint stiffness

To-do:

stopped:
- test slack force (use grav (temp) to slack it)
- apply force
"""

from time import time
# import math
import numpy as np
import der_mujoco.utils.transform_utils as T
import der_mujoco.controllers.der_cpp.Der_iso as Der_iso
# from der_mujoco.utils.filters import ButterLowPass
# from der_mujoco.utils.der_utils import ang_btwn3

class DERRopeBase:
    # Base rope controller is for both ends fixed
    def __init__(
        self,
        sim,
        n_link,
        overall_rot=0.,
        alpha_bar=1.345/10,
        beta_bar=0.789/10,
        damp_const=0.,
        f_limit=1000.,  # not used anymore (change alpha beta to tune)
    ):
        self.sim = sim

        # init variable
        self.d_vec = 0
        self.nv = n_link - 1 - self.d_vec * 2
        self.vec_siteid = np.zeros(self.nv+2, dtype=int)
        self.vec_bodyid = np.zeros(self.nv+2, dtype=int)
        self.link_bodyid = np.zeros(self.nv+1, dtype=int)
        
        self._init_sitebody()

        self.x = np.zeros((self.nv+2,3))
        self.x_vel = np.zeros((self.nv+2,3))
        self.e = np.zeros((self.nv+1,3))

        self.force_node = np.zeros((self.nv+2,3))
        self.force_node_prev = np.zeros((self.nv+2,3))
        self.force_node_flat = self.force_node.flatten()

        self.reset_rot = overall_rot
        self.overall_rot = self.reset_rot
        self.p_thetan = self.reset_rot % (2. * np.pi)
        if self.p_thetan > np.pi:
            self.p_thetan -= 2 * np.pi
        self.theta_displace = self.p_thetan

        # Calc bars
        self.e_bar = np.zeros(self.nv+1)

        # define variable constants
        self.alpha_bar = alpha_bar
        self.beta_bar = beta_bar
        self.f_limit = f_limit
        
        # define base bishop frame
        self.bf0_bar = np.zeros((3,3))
        self.bf_end = np.zeros((3,3))
        self.bf_end_flat = self.bf_end.flatten()

        # rope dynamics parameters
        self.damp_const = damp_const

        self.der_joint_ids = []
        self.der_joint_qveladdr = []
        for i in range(2, n_link):    # from freejoint2 to freejoint(n_links)
            fj_str = 'freejoint'+str(i)
            self.der_joint_ids.append(self.sim.model.joint_name2id(
                fj_str
            ))
            self.der_joint_qveladdr.append(
                self.sim.model.get_joint_qvel_addr(
                    fj_str
                )
            )
        self.der_joint_qveladdr = np.array(self.der_joint_qveladdr)
        self.rotx_qveladdr = self.der_joint_qveladdr[:,0] + 3
        self.rxqva_len = len(self.rotx_qveladdr)

        self._init_der_cpp()

        # init filter
        # fs = 1.0 / self.sim.model.opt.timestep
        # cutoff = 30
        # self.lowpass_filter = ButterLowPass(cutoff, fs, order=5)

    def _init_sitebody(self):
        for i in range(self.d_vec, self.nv+2 + self.d_vec):
            ii = (self.nv+1 + 2*self.d_vec) - i  
            # id starts from 'last' section
            self.vec_siteid[i - self.d_vec] = self.sim.model.site_name2id(
                'r_joint{}_site'.format(ii)
            )
            self.vec_bodyid[i - self.d_vec] = self.sim.model.body_name2id(
                'r_joint{}_body'.format(ii)
            )
            if ii > self.d_vec:
                self.link_bodyid[i - self.d_vec] = (
                    self.sim.model.body_name2id(
                        'r_link{}_body'.format(ii)
                    )
                )
        #     print('r_joint{}_body'.format(ii))
        # input()
        # for i in range(self.nv+1):
        #     print(self.sim.model.body_id2name(self.link_bodyid[i]))
        # print(self.sim.data.body_xpos[
        #     self.link_bodyid[self.nv],:
        # ])
        # input()
        self.link_bodyid[self.nv - self.d_vec] = (
            self.sim.model.body_name2id(
                'stiffrope'
            )
        )
        self.startsec_site = self.sim.model.site_name2id(
            'r_joint{}_site'.format(self.nv+1 + self.d_vec)
        )
        self.endsec_site = self.sim.model.site_name2id(
            'r_joint{}_site'.format(1 + self.d_vec)
        )

    def _update_xvecs(self):
        # start_t = time()

        self.x = self.sim.data.site_xpos[
            self.vec_siteid[:]
        ].copy()
        # end_t1 = time()

        # for i in range(self.nv+2):
        #     # self.x_prev[i] = self.x[i]
        #     self.x[i] = self.sim.data.site_xpos[self.vec_siteid[i]].copy()
        #     # self.x_vel[i] = self.x[i] - self.x_prev[i]
        #     # print(self.sim.data.site_xvelp[self.vec_siteid[i]])
        #     # print(self.sim.model.site_id2name(self.vec_siteid[i]))
        #     # input(self.x[i])

        # end_t2 = time()
        # print(f"t_np = {end_t1 - start_t}")
        # print(f"t_loop = {end_t2 - end_t1}")

    def _init_resetbody_vars(self):
        self.xpos_reset = self.sim.model.body_pos[
            self.link_bodyid[:]
        ].copy()
        self.xquat_reset = self.sim.model.body_quat[
            self.link_bodyid[:]
        ].copy()
        # input(self.xpos_reset)

    def _init_der_cpp(self):
        self._update_xvecs()
        self._init_resetbody_vars()
        self._x2e()
        self._init_bf()
        # self._update_bishf()
        self.der_math = Der_iso.DER_iso(
            self.x.flatten(),
            self.bf0_bar.flatten(),
            self.p_thetan,
            self.overall_rot,
            self.alpha_bar,
            self.beta_bar
        )
        self._init_o2m()

    def _update_der_cpp(self):
        # t1 = time()
        self._update_xvecs()
        # t2 = time()
        self._update_bishf_S()
        # t3 = time()
        bf_align = self.der_math.updateVars(
            self.x.flatten(),
            self.bf0_bar.flatten(),
            self.bf_end_flat
        )
        # t4 = time()
        self.bf_end = self.bf_end_flat.reshape((3,3))
        # t5 = time()
        self.p_thetan = self._get_thetan()
        # t6 = time()
        self.overall_rot = self.der_math.updateTheta(self.p_thetan)
        # print(f"self.p_thetan = {self.p_thetan}")
        # print(f"self.overall_rot = {self.overall_rot}")
        # input()
        # t7 = time()
        # print(f"s1 = {(t2 - t1) * 51.}")
        # print(f"s2 = {(t3 - t2) * 51.}")
        # print(f"s3 = {(t4 - t3) * 51.}")
        # print(f"s4 = {(t5 - t4) * 51.}")
        # print(f"s5 = {(t6 - t5) * 51.}")
        # print(f"s6 = {(t7 - t6) * 51.}")
        return bf_align

    def get_dersim(self):
        ropeend_pos = self.sim.model.body_pos[
            self.link_bodyid[-1],:
        ].copy()
        ropeend_quat = self.sim.model.body_quat[
            self.link_bodyid[-1],:
        ].copy()
        return (
            ropeend_pos,
            ropeend_quat,
            self.overall_rot,
            self.p_thetan
        )

    def set_dersim(
        self,
        ropeend_pos,
        ropeend_quat,
        overall_rot,
        p_thetan
    ):
        self.sim.model.body_pos[
            self.link_bodyid[-1],:
        ] = ropeend_pos
        self.sim.model.body_quat[
            self.link_bodyid[-1],:
        ] = ropeend_quat
        self.overall_rot = overall_rot
        self.p_thetan = p_thetan
        self.der_math.resetTheta(self.p_thetan, self.overall_rot)

    def reset_body(self):
        # input(self.xpos_reset)
        self.sim.model.body_pos[
            self.link_bodyid[:],:
        ] = self.xpos_reset.copy()
        self.sim.model.body_quat[
            self.link_bodyid[:],:
        ] = self.xquat_reset.copy()
        self._reset_vel()
        self.sim.forward()

    def reset_sim(self):
        self.overall_rot = self.reset_rot
        self.p_thetan = self.reset_rot % (2. * np.pi)
        if self.p_thetan > np.pi:
            self.p_thetan -= 2 * np.pi
        self.der_math.resetTheta(self.p_thetan, self.overall_rot)

    def change_ropestiffness(self, alpha_bar, beta_bar):
        self.alpha_bar = alpha_bar  # bending modulus
        self.beta_bar = beta_bar    # twisting modulus
        self.der_math.changeAlphaBeta(self.alpha_bar, self.beta_bar)

    # ~~~~~~~~~~~~~~~~~~~~~~|formula functions|~~~~~~~~~~~~~~~~~~~~~~


    def _x2e(self):
        self.bigL_bar = 0.
        # includes calculations for e_bar
        self.e[0] = self.x[1] - self.x[0]
        self.e_bar[0] = np.linalg.norm(self.e[0])
        for i in range(1,self.nv+1):
            self.e[i] = self.x[i+1] - self.x[i]
            self.e_bar[i] = np.linalg.norm(self.e[i])

    def _init_bf(self):
        parll_tol = 1e-6
        self.bf0_bar[0,:] = self.e[0] / self.e_bar[0]
        self.bf0_bar[1,:] = np.cross(self.bf0_bar[0,:], np.array([0, 0, 1.]))
        if np.linalg.norm(self.bf0_bar[1,:]) < parll_tol:
            self.bf0_bar[1,:] = np.cross(self.bf0_bar[0,:], np.array([0, 1., 0]))
        self.bf0_bar[1,:] /= np.linalg.norm(self.bf0_bar[1,:])
        self.bf0_bar[2,:] = np.cross(self.bf0_bar[0,:], self.bf0_bar[1,:])

    def _update_bishf_S(self):
        # updates start of bishop frame
        # t1 = time()
        mat_res = np.zeros(9)
        self.der_math.calculateOf2Mf(
            self.sim.data.site_xmat[self.startsec_site],
            mat_res
        )
        mat_res = mat_res.reshape((3,3))
        self.bf0_bar = np.transpose(mat_res)
        # self.bf0_bar = np.transpose(self._of2mf(
        #     self.sim.data.site_xmat[self.startsec_site].reshape((3, 3))
        # ))
        # t3 = time()
        # print(f"*total = {(t3 - t1) * 51.}")
    def _of2mf(self, mat_o):
        # converts object frame to material frame
        return T.quat2mat(
            T.quat_multiply(
                T.mat2quat(mat_o),
                self.qe_o2m_loc
            )
        )

    def _mf2of(self, mat_m):
        # converts material frame to object frame
        return T.quat2mat(
            T.quat_multiply(
                T.mat2quat(mat_m),
                self.qe_m2o_loc
            )
        )

    def _init_loc_rotframe(self, q1, q2):
        qe = T.axisangle2quat(T.quat_error(q1, q2))
        q1_inv = T.quat_inverse(q1)
        qe_loc = T.quat_multiply(T.quat_multiply(q1_inv,qe),q1)
        return qe_loc
    
    def _init_o2m(self):
        # init for mf_adapt and p_thetan
        q_o0 = T.mat2quat(
            self.sim.data.site_xmat[self.startsec_site].reshape((3, 3))
        )
        q_b0 = T.mat2quat(
            np.transpose(self.bf0_bar)
        )
        self.qe_o2m_loc = self._init_loc_rotframe(q_o0, q_b0)
        self.qe_m2o_loc = self._init_loc_rotframe(q_b0, q_o0)

        self.der_math.initQe_o2m_loc(self.qe_o2m_loc)

    def _get_thetan(self):
        # t1 = time()
        mat_on = self.sim.data.site_xmat[self.endsec_site]
        # t2 = time()
        mat_bn = np.transpose(self.bf_end)
        # t3 = time()
        mat_mn = np.zeros(9)
        self.der_math.calculateOf2Mf(
            mat_on,
            mat_mn
        )
        mat_mn = mat_mn.reshape((3,3))
        # mat_mn = self._of2mf(mat_on)
        # input()
        # if np.linalg.norm(mat_bn[:,0] - mat_mn[:,0]) > 1e-6:
        #     raise Exception(
        #         f"""
        #         x-axis of mf and bf not aligned.\n
        #         of = {mat_on}
        #         bf = {mat_bn}\n
        #         mf = {mat_mn}\n
        #         """
        #     )
        # print(mat_mn)
        # print(mat_bn)
        # print(f"thetanraw = {self.der_math.angBtwn3(mat_bn[:,1], mat_mn[:,1], mat_bn[:,0])}")
        # print(f"sim_thetan = {theta_n}")
        # if math.isnan(theta_n): input()
        # t4 = time()
        theta_n = (
            self.der_math.angBtwn3(mat_bn[:,1], mat_mn[:,1], mat_bn[:,0])
            + self.theta_displace
        )
        # theta_n = (
        #     ang_btwn3(mat_bn[:,1], mat_mn[:,1], mat_bn[:,0])
        #     + self.theta_displace
        # )
        # t5 = time()
        # print(f"**s1 = {(t2 - t1) * 51.}")
        # print(f"**s2 = {(t3 - t2) * 51.}")
        # print(f"**s3 = {(t4 - t3) * 51.}")
        # print(f"**s4 = {(t5 - t4) * 51.}")
        return theta_n

    def _calc_centerlineF(self):
        # self.force_node_flat = np.zeros((self.nv+2)*3)
        self.der_math.calculateCenterlineF2(self.force_node_flat)
        self.force_node = self.force_node_flat.reshape((self.nv+2,3))

    def _damp_force(self, excl_joints):
        # damp forces based on velocity of nodes
        
        # t1 = 0.

        # print(self.sim.data.body_xvelp[
        #             self.vec_bodyid[excl_joints:self.nv+1-excl_joints]
        #     ]
        # )
        # print(self.e_bar[excl_joints:(self.nv+1-excl_joints), np.newaxis])
        # print('d')
        # input()
        # start_t = time()
        self.force_node[excl_joints:self.nv+1-excl_joints] -= (
            np.multiply(
                self.sim.data.body_xvelp[
                    self.vec_bodyid[excl_joints:self.nv+1-excl_joints]
                ],
                self.e_bar[excl_joints:(self.nv+1-excl_joints), np.newaxis]
            )
        ) * self.damp_const

        # end_t1 = time()

        # t1 = (end_t1 - start_t)
        # print(f"time np = {t1}")

    def _limit_force(self, f1):
        # limit the force on each node to self.f_limit
        for i in range(len(f1)):
            f1_mag = np.linalg.norm(f1[i])
            if f1_mag > self.f_limit:
                f1[i] *= self.f_limit / f1_mag
        #         print(self.f_limit)
        #         print(f1_mag)
        #         print(np.linalg.norm(f1[i]))
        #         input(f1[i])
        # print(f1)
        # return f1

    def _limit_totalforce(self):
        # limit the total force magnitude of all nodes to self.f_limit
        self.f_limit = 30.
        force_mag = np.linalg.norm(self.force_node)
        if force_mag > self.f_limit:
            self.force_node *= self.f_limit / force_mag
        # print(self.force_node)
        # input()

    # def _limit_forcechange(self):
    #     # print('here++++++++++++++++++++++++++++++++++++++')
    #     for i in range(len(self.force_node)):
    #         f_diff = np.linalg.norm(
    #             self.force_node[i] - self.force_node_prev[i]
    #         )
    #         if f_diff > 5:
    #             input(f_diff)
    #     self.force_node_prev = self.force_node.copy()

    def _reset_vel(self):
        self.sim.data.body_xvelp[self.link_bodyid[:]] = np.zeros(
            (self.nv+1,3)
        )
        self.sim.data.body_xvelr[self.link_bodyid[:]] = np.zeros(
            (self.nv+1,3)
        )

    def reset_qvel_rotx(self):
        # t1 = np.array([12,18,24])
        # t2 = np.zeros(len(t1))
        # self.sim.data.qvel[t1] = t2
        self.sim.data.qvel[self.rotx_qveladdr] = np.zeros(self.rxqva_len)
        # self.sim.forward()

    def get_ropevel(self):
        return self.sim.data.body_xvelp[self.link_bodyid[:]]

    def update_force(self, f_scale=1.0, limit_f=False):
        # start_t = time()
        bf_align = self._update_der_cpp()
        # print(f"bf_align = {bf_align} ++++++++++++++++++++++++")
        # end_t1 = time()
        excl_joints = 2 - 2*self.d_vec # 2 joints excluded from each side (no force)
        self._calc_centerlineF()
        # end_t2 = time()

        # self._damp_force(excl_joints)
        self.avg_force = np.linalg.norm(self.force_node)/len(self.force_node)
        # print(f"avg_force = {np.linalg.norm(self.force_node)/len(self.force_node)}")
        if limit_f:
            self._limit_totalforce()
        # end_t3 = time()
        
        # if True:
        #     self.force_node = self.lowpass_filter(
        #         self.force_node.reshape((-1,3*len(self.force_node)))
        #     )[0,:].reshape((-1,3))
        # if True:    # helps 20
        #     self.n_fn = self.force_node.copy()
        #     self.force_node = (self.n_fn + self.p_fn) / 2
        #     self.p_fn = self.force_node.copy()  # using avg as p
        #     # self.p_fn = self.n_fn.copy()    # using prev as p
        # print(self.force_node)
        # f_scale = 1.0 # 0.2 # 9e-5
        # print(self.force_node)
        # print(self.sim.data.qvel[self.rotx_qveladdr])
        # print(self.sim.data.ctrl[:])
        # print(self.sim.model.joint_name2id(3))
        # print(self.sim.model.get_joint_qvel_addr('freejoint3'))
        # print('here')
        # input()
        # print(self.sim.data.body_xvelr[self.link_bodyid[:]])
        # print(f"self.force_node = {self.force_node}")
        ids_i = range(excl_joints,self.nv+2-excl_joints)
        # for i in ids_i:
        #     print(f"body_ids = {self.sim.model.body_id2name(self.vec_bodyid[i])}")
        # print(f"force_node = {self.force_node[ids_i]}")
        # input()
        # self._limit_forcechange()
        # self._limit_force(self.force_node)
        # input(self.force_node)
        self.sim.data.xfrc_applied[self.vec_bodyid[ids_i],:3] = (
            f_scale * self.force_node[ids_i]
        )
        # self.reset_qvel_rotx()
        # for i in range(excl_joints,self.nv+2-excl_joints):
            # # self.force_node[i] = self._limit_force(self.force_node[i])
            # self.sim.data.xfrc_applied[self.vec_bodyid[i],:3] = (
                # f_scale * self.force_node[i]
            # )
        # self._reset_vel()
        # end_t4 = time()

        # print(f"t_update = {end_t1 - start_t}")
        # print(f"t_calcF = {end_t2 - end_t1}")
        # print(f"t_aug_F = {end_t3 - end_t2}")
        # print(f"t_apply_F = {end_t4 - end_t3}")

        # print(f"overall0 = {end_t4 - start_t}")

        return self.force_node.copy(), self.x.copy(), bf_align
        
    # def calc_E_bend(self):
    #     self.ee_bend = 0.
    #     for i in range(1,self.nv+1):
    #         sum_temp = 0.
    #         sum_temp += (
    #             self.alpha_bar
    #             * np.dot(self.kb[i], self.kb[i])
    #         )
    #         sum_temp /= self.l_bar[i]
    #         self.ee_bend += sum_temp
    #     return self.ee_bend
    
    # def calc_E_twist(self):
    #     self.ee_twist = 0.
    #     for i in range(1,self.nv+1):
    #         m = self.theta[i] - self.theta[i-1]   # discrete twist
    #         self.ee_twist += (
    #             self.beta_bar * (m**2)
    #             / self.l_bar[i]
    #         )
    #     return self.ee_twist

# ~~~~~~~~~~~~~~~~~~~~~~~~|End of Class|~~~~~~~~~~~~~~~~~~~~~~~~


#     # Calc and apply force (might need scaling)

# class DERRope1(DERRopeBase):
#     def __init__(self, sim, n_vec, overall_rot=0):
#         super().__init__(sim, n_vec, overall_rot)

#     def _init_sitebody(self):
#         for i in range(self.d_vec, self.nv+2 + self.d_vec):  # id starts from 'first' section
#             self.vec_siteid[i - self.d_vec] = self.sim.model.site_name2id(
#                 'r_joint{}_site'.format(i)
#             )
#             self.vec_bodyid[i - self.d_vec] = self.sim.model.body_name2id(
#                 'r_joint{}_body'.format(i)
#             )
#             if i > self.d_vec:
#                 self.link_bodyid[i - self.d_vec] = (
#                     self.sim.model.body_name2id(
#                         'r_link{}_body'.format(i)
#                     )
#                 )
#         self.startsec_site = self.sim.model.site_name2id(
#             'r_link{}_site'.format(1 + self.d_vec)
#         )
#         self.endsec_site = self.sim.model.site_name2id(
#             'r_link{}_site'.format(self.nv+1 + self.d_vec)
#         )

#     def _update_theta_n(self):
#         self.theta[-1] = 0.
#         # print(f"theta_n = {0.}")
#         # print(self.theta[-1])