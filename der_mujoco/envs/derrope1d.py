#!/usr/bin/env python3

from time import time
import numpy as np
import matplotlib.pyplot as plt
import os
import mujoco_py
import gym
from gym import utils, spaces
import pickle

import der_mujoco.utils.transform_utils as T
from der_mujoco.utils.transform_utils import IDENTITY_QUATERNION
from der_mujoco.utils.mjc_utils import MjSimWrapper
from der_mujoco.utils.xml_utils import XMLWrapper
from der_mujoco.utils.data_utils import compute_PCA
from der_mujoco.assets.gen_derrope import DERGen_eq1D, DERGen_2D
from der_mujoco.controllers.rope_controller import DERRopeBase


IDENTITY_QUATERNION = np.array([1.0, 0, 0, 0])

class DerRope1DEnv(gym.Env, utils.EzPickle):

    def __init__(
        self,
        do_render=True,
        both_fixed=True,
        r_pieces=40,    # max. 33
    ):
        """
        This is an environment to move the rope from the left of an obstacle
        (view up-down as fixed box to eef of robot, respectively) 
        to the right. Done by checking where the rope intersects
        the same y_pos value as the obstacle. 
        
        self.y_crossed shows the state:
        - True: crossed over to the right side
        - False: still on the left side

        Env aims to achieve self.y_crossed=True for as long as possible
        in the 1000 env_steps window.

        Forces and torques:
        - world frame x, y, and z.
        - expressed as arm acting on rope
        """
        utils.EzPickle.__init__(self)

        # simulation-specific attributes
        self.viewer = None
        
        # self.model = None
        self.sim = None
        self.do_render = do_render

        # rope params init
        self.r_pieces = r_pieces

        # load model
        # update rope model
        world_base_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "assets/world.xml"
        )
        rope_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "assets/derrope1d.xml"
        )
        box_path = os.path.join(
            os.path.dirname(rope_path),
            "anchorbox.xml"
        )
        # self.r_len = 9.29
        self.r_len = 2*np.pi * self.r_pieces / (self.r_pieces-1)
        self.r_thickness = 0.05
        init_pos = [0., 0., 0.50]
        init_pos[0] += self.r_len
        DERGen_eq1D(
            r_len=self.r_len,
            r_thickness=self.r_thickness,
            r_pieces=self.r_pieces,
            j_stiff=0.0,
            j_damp=0.0,
            init_pos=init_pos,
            d_small=0.,
            both_fixed=both_fixed,
            rope_type="cylinder",
            vis_subcyl=False,
            obj_path=rope_path,
        )
        self.xml = XMLWrapper(world_base_path)
        derrope = XMLWrapper(rope_path)
        anchorbox = XMLWrapper(box_path)
        self.xml.merge_multiple(
            anchorbox, ["worldbody", "equality", "contact"]
        )
        self.xml.merge_multiple(
            derrope, ["worldbody"]
        )
        xml_string = self.xml.get_xml_string()

        # initialize simulation
        self.model = mujoco_py.load_model_from_xml(xml_string)
        self.sim = mujoco_py.MjSim(self.model)
        self.sim = MjSimWrapper(self.sim)   # wrap sim
        self.data = self.sim.data
        self.viewer = None

        # init gravity
        self.model.opt.gravity[-1] = 0.

        # rope stuff
        self.rope_site_idx = np.zeros(self.r_pieces, dtype=int)
        for i_sec in range(self.r_pieces):
            self.rope_site_idx[i_sec] = self.sim.model.site_name2id(
                'r_link{}_site'.format(i_sec+1)
            )
        self.joint_site_idx = np.zeros(self.r_pieces+1, dtype=int)
        for i_sec in range(self.r_pieces+1):
            self.joint_site_idx[i_sec] = self.sim.model.site_name2id(
                'r_joint{}_site'.format(i_sec)
            )
        self.eef_site_idx = self.sim.model.site_name2id('r_joint{}_site'.format(
            self.r_pieces
        ))
        
        # initialize viewer
        marker_id = 0 # self.sim.model.body_name2id('marker')
        self.viewer_params = [
            marker_id, 0., 0., 20, 0.9, -30
        ]
        if self.do_render and self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim.sim)
            self.viewer.cam.trackbodyid = self.viewer_params[0]
            self.viewer.cam.lookat[2] -= self.viewer_params[1]
            # self.viewer.cam.lookat[0] += self.viewer_params[2]
            self.viewer.cam.azimuth = self.viewer_params[3]
            self.viewer.cam.distance = (
                self.sim.model.stat.extent
                * self.viewer_params[4]
            ) 
            self.viewer.cam.elevation = self.viewer_params[5]

        # init obs
        self.observations = dict(
            r_eef=np.zeros(3),
            r_pos=np.zeros((self.r_pieces,3)),
        )

        # other variables
        self.max_env_steps = 10000
        self.env_steps = 0
        self.cur_time = 0
        self.dt = self.sim.model.opt.timestep

        # init der controller
        self.step0()
        self.f_limit = 10
        self.overall_rot = 0. # 27 * (2*np.pi) # 57 * (np.pi/180)
        self.alpha_bar = 1.
        self.beta_bar = 1.
        if both_fixed:
            self.der_sim = DERRopeBase(
                sim=self.sim,
                n_link=self.r_pieces,
                alpha_bar=self.alpha_bar,
                beta_bar=self.beta_bar,
                overall_rot=self.overall_rot,
                f_limit=self.f_limit,
            )
        # else:
        #     self.der_sim = DERRope1(
        #         sim=self.sim,
        #         n_vec=self.r_pieces,
        #         overall_rot=0.,
        #         f_limit=self.f_limit,
        #     )
        # self.d_vec = self.der_sim.d_vec

        # self.time_a = 0.

        self.stablestep = False

        # self.circle_init()
        self.start_circletest()
        # self.start_lhbtest()

    def _get_observations(self):
        # rope stuff
        self.observations['r_pos'][:] = np.array(
            self.sim.data.site_xpos[self.rope_site_idx[:]]
        ).copy()
        self.observations["r_eef"] = np.array(
            self.sim.data.site_xpos[
                self.eef_site_idx
            ]
        )
        return np.concatenate(
            (
                self.observations["r_eef"],
            )
        )

    def reset(self):
        self.sim.reset()
        
        if self.do_render and self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim.sim)
            # self.viewer.cam.fixedcamid = 0
            # self.viewer.cam.type = const.CAMERA_FREE
            self.viewer.cam.trackbodyid = self.viewer_params[0]
            self.viewer.cam.lookat[2] -= self.viewer_params[1]
            # self.viewer.cam.lookat[0] += self.viewer_params[2]
            self.viewer.cam.azimuth = self.viewer_params[3]
            self.viewer.cam.distance = (
                self.sim.model.stat.extent
                * self.viewer_params[4]
            ) 
            self.viewer.cam.elevation = self.viewer_params[5]
        # self.viewer._paused = True

        self.der_sim.reset_body()
        self.sim.forward()

        # reset obs
        self.observations = dict(
            r_eef=np.zeros(3),
            r_pos=np.zeros((self.r_pieces,3)),
        )

        # reset der_cpp
        self.der_sim.reset_sim()

        # reset time
        self.cur_time = 0   #clock time of episode
        self.env_steps = 0

        return self._get_observations()

    def render(self):
        self.viewer.render()

    # def _check_proximity(self, pos1, pos2):
    #     # True if close, False if not close enough
    #     distbtwn = pos1 - pos2
    #     if(
    #         np.abs(distbtwn)[2] < 2e-3
    #         and np.linalg.norm(distbtwn[:2]) < 1e-5
    #     ):
    #         return True
    #     return False

    def step0(self):
        self.sim.step()
        self.sim.forward()
        
        if self.do_render:
            self.render()
        self.cur_time += self.dt

    # def _reset_rope_vel(self.)

    def update_force(self):
        # start_t = time()
        bf_align = self.der_sim.update_force()
        if not bf_align: 
            input('bf_align')
            self.reset()
        # for i in range(2 - self.d_vec,self.r_pieces-1 - self.d_vec):
        #     mag_force = np.linalg.norm(node_force[i])
        #     self.viewer.add_marker(
        #         pos=np.array(
        #             [node_pos[i,0], node_pos[i,1], node_pos[i,2]]
        #         ), #position of the arrow
        #         size=np.array([0.005,0.005,mag_force/self.f_limit/5]), 
        #         #size of the arrow
        #         mat=T.vec2randmat(node_force[i]), # orientation as a matrix
        #         rgba=np.array(
        #             [mag_force/self.f_limit,1-mag_force/self.f_limit,0.,0.35]
        #         ),#color of the arrow
        #         type=const.GEOM_ARROW,
        #         # label=str('GEOM_ARROW')
        #     )
        # end_t2 = time()
        # print(f"t_controller = {end_t1 - start_t}")
        # print(f"t_apply = {end_t2 - end_t1}")
        # print(f"overall0.5 = {end_t2 - start_t}")
        # input()

    def reset_vel(self):
        self.data.qvel[:] = np.zeros(len(self.data.qvel[:]))
        self.sim.forward()

    def step(self):
        # vel_dir = np.array([0., 0., 0.])
        # self.give_vel(body_name='anchor_box',vel_dir=vel_dir)

        start_t = time()
        self.update_force()
        end_t1 = time()
        # if self.stablestep and self.env_steps % 10:
        #     self.reset_vel()
        # j_name = self.sim.model.joint_id2name(1)
        # print(j_name)
        # print(self.data.get_joint_qvel(j_name))
        # print(self.data.qvel[:])
        self.sim.step()
        self.sim.forward()
        end_t2 = time()
        if self.do_render:
            self.render()
        self.cur_time += self.dt
        self.env_steps += 1
        end_t3 = time()
        # print(f"overall1 = {end_t1 - start_t}")
        # print(f"stepforward_t = {end_t2 - end_t1}")
        # self.time_a += end_t2 - end_t1
        # print(f"render_t = {end_t3 - end_t2}")
        # input()
            
    def hold_pos(self, hold_time=2.):
        # No env step increment added
        init_time = self.cur_time
        while (self.cur_time-init_time) < hold_time:
            self.step()
            # self.viewer._paused = True
        print(f"time = {self.cur_time}")
        print(f"theta_n (rad) = {self.der_sim.overall_rot}")
        print(f"theta_n (deg) = {self.der_sim.overall_rot*180/np.pi}")

    def ropeend_rot(self, rot_a=np.pi/180, rot_axis=1):
        body_id = self.sim.model.body_name2id("stiffrope")
        rot_arr = np.zeros(3)
        rot_arr[rot_axis] = rot_a
        rot_quat = T.axisangle2quat(rot_arr)
        new_quat = T.quat_multiply(rot_quat, self.model.body_quat[body_id])
        self.model.body_quat[body_id] = new_quat
        self.hold_pos(0.02)
        # t1 = self.der_sim.theta[-1]
        # print(t1)
        # if t1 > (np.pi-0.1) or t1 < 0:
        #     print(self.der_sim.bf_rot)
        #     print(self.der_sim.mf_rot)
        #     print(self.der_sim.theta_n)
        #     input()

    def ropeend_pos(self, pos_move=np.array([0., -1e-4, 0.])):
        body_id = self.sim.model.body_name2id("stiffrope")
        self.model.body_pos[body_id][:] += pos_move.copy()
        self.hold_pos(0.1)

    def apply_force(
        self,
        body_name=None,
        force_dir=np.array([0., 0., 0.01]),
    ):
        if body_name is None:
            body_name = "r_link{}_body".format(int(self.r_pieces/2)+1)
        # print(self.sim.data.xfrc_applied[i])
        body_id = self.sim.model.body_name2id(body_name)
        # print(self.model.body_pos[body_id])
        # print(self.sim.data.xfrc_applied[body_id,:3])
        self.sim.data.xfrc_applied[body_id,:3] = force_dir
        # print(self.sim.data.xfrc_applied[body_id,:3])

    def sitename2pos(self, site_name):
        site_id = self.sim.model.site_name2id(site_name)
        site_pos = self.sim.data.site_xpos[site_id]
        return site_pos

    def bodyname2pos(self, body_name):
        body_id = self.sim.model.body_name2id(body_name)
        return self.model.body_pos[body_id].copy()

    def give_vel(
        self,
        body_name=None,
        vel_dir=np.array([0., 0., 0.01]),
    ):
        if body_name is None:
            body_name = "r_link{}_body".format(int(self.r_pieces/2)+1)
        # print(self.sim.data.xfrc_applied[i])
        body_id = self.sim.model.body_name2id(body_name)
        # print(self.model.body_pos[body_id])
        # print(self.sim.data.xfrc_applied[body_id,:3])
        self.sim.data.body_xvelp[body_id] = vel_dir
        print(vel_dir)
        print(self.sim.data.body_xvelp[body_id])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| Pickle things ||~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_state(self):
        (
            ropeend_pos,
            ropeend_quat,
            overall_rot,
            p_thetan
        ) = self.der_sim.get_dersim()
        return [
            np.concatenate((
                [0], # [self.cur_time],
                [0], # [self.env_steps],
                ropeend_pos,
                ropeend_quat,
                [overall_rot],
                [p_thetan],
            )),
            self.sim.sim.get_state()
        ]

    def set_state(self, p_state):
        self.cur_time = 0 # p_state[0][0]
        self.env_steps = 0 # p_state[0][1]
        self.der_sim.set_dersim(
            p_state[0][2:5],
            p_state[0][5:9],
            p_state[0][9],
            p_state[0][10],
        )
        self.sim.sim.set_state(p_state[1])
        self._get_observations()
        self.der_sim._update_xvecs()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| Testing things ||~~~~~~~~~~~~~~~~~~~~~~~~~~

    def circle_init(self):
        # assuming that rope axis is parallel to x axis
        self.stablestep = False
        y_offset = 0.3
        start_pos = self.bodyname2pos('stiffrope')
        end_pos = self.bodyname2pos(f'r_link{self.r_pieces}_body')
        e_pos = end_pos - start_pos
        step_len = 5e-3
        n_steps = e_pos[0] / step_len
        if n_steps < 0:
            step_len = - step_len
            n_steps = int(-n_steps)
        step_ylen = 2 * y_offset / n_steps
        n_steps = int(n_steps/2)
        step_remain = e_pos[0] - 2* n_steps * step_len
        for i in range(n_steps):
            self.ropeend_pos(np.array([step_len, step_ylen, 0.]))
        for i in range(360):
            self.ropeend_rot(rot_axis=2)
        for i in range(n_steps):
            self.ropeend_pos(np.array([step_len, 0., 0.]))
        self.ropeend_pos(np.array([step_remain - 1e-3, 0., 0.]))
        for i in range(int(n_steps)):
            self.ropeend_pos(np.array([0., -step_ylen, 0.]))
        
        self.stablestep = False
        self.hold_pos()
        self.reset_vel()
        self.hold_pos()
        for i in range(50):
            self.hold_pos(0.3)
            self.reset_vel()

    def check_e_PCA_circle(self):
        joint_site_pos = np.array(
            self.sim.data.site_xpos[self.joint_site_idx[:]]
        ).copy()
        joint_site_pos = joint_site_pos[:self.r_pieces-1]
        # exclude 2 nodes to account for overlap
        return compute_PCA(
            joint_site_pos[:,0],
            joint_site_pos[:,1],
            joint_site_pos[:,2],
        )

    def start_circletest(self):
        # Test for multiple rope types (vary alpha and beta bar)
        # alpha = 1.
        # r_len = 2*np.pi * self.r_pieces / (self.r_pieces-1)
        new_start = False
        e_tol = 1000. # 0.075
        if new_start:
            self.circle_init()
            self.init_pickle = self.get_state()
            with open('circtest2.pickle', 'wb') as f:
                pickle.dump(self.init_pickle,f)
            input('Pickle saved!')
        else:
            with open('circtest2.pickle', 'rb') as f:
                self.init_pickle = pickle.load(f)

            # set overall rot
            self.init_pickle[0][9] = self.overall_rot
            self.p_thetan = self.overall_rot % (2. * np.pi)
            if self.p_thetan > np.pi:
                self.p_thetan -= 2 * np.pi
            self.init_pickle[0][10] = self.p_thetan

            self.set_state(self.init_pickle)
            self.hold_pos(10.)
        
        self.apply_force(force_dir=np.array([0., 0., 1.]))
        self.hold_pos(0.5)
        self.apply_force(force_dir=np.array([0., 0., 0.]))
        # self.hold_pos(100.)
        # create a for loop that checks PCA error at each iter
        for i in range(5000):
            # self.ropeend_rot(rot_axis=0)
            # if not i % int(1.76*360) and i != 0:
            #     self.reset_vel()
            #     self.hold_pos(10.)
            # if not i % 10:
            #     self.reset_vel()
            #     self.hold_pos(0.3)
            # self.reset_vel()
            self.hold_pos(0.3)
            e_outofplane = self.check_e_PCA_circle()
            print(f'angle twisted = {i*np.pi/180}')
            print(f'e_outofplane = {e_outofplane}')
            if e_outofplane > e_tol:
                print(i * np.pi / 180)
                input(e_outofplane)
                break
        return e_outofplane

    def lhb_init(self):
        # Twist 27 turns (27 * 360)
        l_shorten = 0.3
        n_steps = 100
        step_len = l_shorten / n_steps
        # n_steps = int(l_shorten / step_len)
        for i in range(2):
            self.ropeend_pos(np.array([0., -step_len, 0.]))
        for i in range(1):
            self.ropeend_pos(np.array([-step_len, 0., 0.]))
        for i in range(2):
            self.ropeend_pos(np.array([0., step_len, 0.]))
        for i in range(n_steps-1):
            self.ropeend_pos(np.array([-step_len, 0., 0.]))
            # self.reset_vel()
        # step_remain = l_shorten - 2* n_steps * step_len
        # self.ropeend_pos(np.array([step_remain, 0., 0.]))
        self.hold_pos(15.)
        for i in range(10):
            self.reset_vel()
            self.hold_pos(.5)

    def start_lhbtest(self):
        # Test for 1 rope type - length, alpha, beta
        # localized helical buckling test
        new_start = True
        if new_start:
            self.lhb_init()
            self.init_pickle = self.get_state()
            with open('lhbtest.pickle', 'wb') as f:
                pickle.dump(self.init_pickle,f)
            input('Pickle saved!')
        else:
            with open('lhbtest.pickle', 'rb') as f:
                self.init_pickle = pickle.load(f)

            # set overall rot
            self.init_pickle[0][9] = self.overall_rot
            self.p_thetan = self.overall_rot % (2. * np.pi)
            if self.p_thetan > np.pi:
                self.p_thetan -= 2 * np.pi
            self.init_pickle[0][10] = self.p_thetan

            self.set_state(self.init_pickle)
            self.hold_pos(0.1)

        joint_site_pos = np.array(
            self.sim.data.site_xpos[self.joint_site_idx[:]]
        ).copy()
        t_vec = joint_site_pos[1:] - joint_site_pos[:self.r_pieces]
        for i in range(len(t_vec)):
            t_vec[i] = t_vec[i] / np.linalg.norm(t_vec[i])
        print(t_vec)
        e_x = np.array([-1., 0., 0.])
        devi_set = np.arccos(np.dot(t_vec, e_x))
        max_devi = np.max (devi_set)
        rot_step = self.der_sim.overall_rot / (self.r_pieces - 1)
        s_step = self.r_len / (self.r_pieces)
        m = np.zeros(len(devi_set))
        s = np.zeros(len(devi_set))
        for i in range(len(devi_set)):
            m[i] = rot_step * i
            s[i] = s_step * i + s_step/2
        s = s - np.mean(s)
        max_devi2 = max_devi
        # max_devi2 = 0.919
        print(rot_step)
        s_ss = (
            s * (self.beta_bar*12/(2*self.alpha_bar))
            * np.sqrt((1-np.cos(max_devi2))/(1+np.cos(max_devi2)))
        )
        # s_ss = s_ss - np.mean(s_ss)
        fphi = (np.cos(devi_set)-np.cos(max_devi))/(1-np.cos(max_devi))
        print(f"fphi = {fphi}")
        print(f"s_ss = {s_ss}")
        print(f"max_devi = {max_devi}")
        pickledata = [fphi, s_ss]
        pickledata_path = 'lhb{}.pickle'.format(self.r_pieces)
        pickledata_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data/" + pickledata_path
        )
        with open(pickledata_path, 'wb') as f:
            pickle.dump(pickledata,f)
        plt.figure("Localized Helical Buckling")
        plt.xlabel("s/s*")
        plt.ylabel(r'$f(\varphi)$')
        plt.plot(s_ss, fphi)
        plt.show()
        input()

        # Bring together by dL units
        # for loop going through each piece
        #   check the vector from one site to the next - then normalize

        # get tangent axis vector (x-axis) 