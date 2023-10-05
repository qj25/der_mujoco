#!/usr/bin/env python3

"""
To-do:
- create traj finder:
    - find sample points
    - do ik
    - create spline curve for each joint
        - (joint angle against time)
        - constant time interval between sample points
        - use appropriate total time (check successful flinging time)
    - create controller to carry out spline curve trajectory
        - control in joint space
    
- change observations:
    - observations is the position of the rope and obs_top 
    wrt the fixed end.
    - allow for adjustments in rl actions in the case of deviation.

Improve:
- change rope mass model

Ideas:
- Make time parameterization and total time an RL variable
"""
from time import time
import numpy as np
import matplotlib.pyplot as plt
from math import floor
import os
import mujoco_py
import gymnasium as gym
# import gym
from gymnasium import utils
# from gymnasium import spaces
import der_mujoco.utils.transform_utils as T
from der_mujoco.utils.transform_utils import IDENTITY_QUATERNION
from der_mujoco.utils.mjc_utils import MjSimWrapper
from der_mujoco.utils.xml_utils import XMLWrapper
from der_mujoco.assets.gen_derrope2 import DERGen_eq1D
# from der_mujoco.assets.gen_misclr import ObsGen
# from der_mujoco.utils.der_utils import ang_btwn2
# from der_mujoco.utils.ik.Franka_ik_He import Fih
# from der_mujoco.utils.spline_utils import spline_traj
# from der_mujoco.utils.spline_utils import _get_relmat
from der_mujoco.utils.data_utils import compute_PCA
import pickle

# from der_mujoco.controllers.rope_controller import DERRopeBase, DERRope1
from der_mujoco.controllers.rope_controller2 import DERRopeBase
# from der_mujoco.controllers.joint_controller import JointController
# from der_mujoco.controllers.joint_controller import joint_sum

IDENTITY_QUATERNION = np.array([1.0, 0, 0, 0])

class DerRope1DEnv(gym.Env, utils.EzPickle):
    def __init__(
        self,
        do_render=True,
        both_fixed=True,
        r_pieces=20,    # max. 33
        r_len = 2*np.pi,
        test_type=None,
        overall_rot=None,
        alpha_bar=1.345/4500,   # 1.345/50,
        beta_bar=0.789/4500,    # 0.789/50,
        r_mass=5.,
        new_start=False,
    ):
        """
        This is an environment to test the rope (validation).
        """
        utils.EzPickle.__init__(self)

        # settings
        self.indiv_steps = False
        self.sample_workspace = True

        # simulation-specific attributes
        self.viewer = None
        
        # self.model = None
        self.sim = None
        self.do_render = do_render
        self.test_type = test_type

        # rope params init
        self.r_len = r_len
        self.r_mass = r_mass
        self.r_pieces = r_pieces
        self.alpha_bar = alpha_bar
        self.beta_bar = beta_bar
        init_pos = [0., 0., 0.50]
        init_pos[0] += self.r_len

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
        
        DERGen_eq1D(
            r_len=self.r_len,
            r_thickness=0.03,
            r_pieces=self.r_pieces,
            r_mass=self.r_mass,
            j_stiff=0.0,
            j_damp=0.0,
            init_pos=init_pos,
            # init_quat=[0.707, 0., 0., -0.707],
            d_small=0.,
            both_fixed=both_fixed,
            rope_type="capsule",
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
        self.xml.merge_multiple(
            derrope, ["sensor"]
        )
        
        xml_string = self.xml.get_xml_string()
        asset_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "assets/overall.xml"
        )
        with open(asset_path, "w+") as f:
            f.write(xml_string)
        # input(xml_string)

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

        self.t_long = 0.3

        # init der controller
        self.zero_step()
        # self.f_limit = 1000.
        # self.damp_const = 0.
        self.overall_rot = 0. # 27 * (2*np.pi) # 57 * (np.pi/180)
        if overall_rot is not None:
            self.overall_rot = overall_rot
        if both_fixed:
            self.der_sim = DERRopeBase(
                sim=self.sim,
                n_link=self.r_pieces,
                alpha_bar=self.alpha_bar,
                beta_bar=self.beta_bar,
                overall_rot=self.overall_rot,
                # f_limit=self.f_limit,
                # damp_const=self.damp_const
            )

        self.d_vec = self.der_sim.d_vec

        self.time_a = 0.

        self._update_ropepos()

        self.n_ropeseg = 3
        self.impt_seg = []
        for i in range(self.n_ropeseg-1):
            self.impt_seg.append(int(self.r_pieces*(i+1)/self.n_ropeseg-0.5))

        if self.test_type == 'lhb':
            self.start_lhbtest(new_start)
        elif self.test_type == 'mbi':
            self.start_circletest(new_start)

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

        rope_pos = self.observations["r_pos"][self.impt_seg]
        return np.float32(
            [self.observations["r_pos"]]   
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        # print('1 iter')
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
        # self.data.ctrl[:] = np.zeros(self.model.nv)
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

        # self._load_initpickle()
        # self.data.qpos[:7] = np.array(self.init_qpos)
        # self.viewer._paused = True
        # print(self.observations['eef_pos'])
        info = dict(
            eps_time=self.cur_time,
            sim_steps=self.cur_time / self.dt,
        )
        return self._get_observations(), info

    def render(self):
        self.viewer.render()

    def zero_step(self):
        self.sim.step()
        self.sim.forward()
        
        # if self.do_render:
        #     self.render()
        self.cur_time += self.dt

    def update_force(self):
        # start_t = time()
        # if self.cur_time < 10.0:
        #     bf_align = self.der_sim.update_force(f_scale=self.cur_time/100.0)
        # else:
        #     bf_align = self.der_sim.update_force()
        bf_align = self.der_sim.update_force()
        if not bf_align: 
            input('bf_align')
            self.reset()

        if self.test_type == 'lhb':
            self.avg_force[0] = (
                self.der_sim.avg_force
                + self.avg_force[0] * self.avg_force[1]
            ) / (self.avg_force[1] + 1)
            self.avg_force[1] += 1
        # end_t1 = time()
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
        start_t = time()
        # if self.cur_time > 3.:
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
            
    def hold_pos(self, hold_time=2.):
        init_time = self.cur_time
        # self.print_collisions()
        while (self.cur_time-init_time) < hold_time:
            # print(f"self.cur_time = {self.cur_time}")
            self.step()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| Other things ||~~~~~~~~~~~~~~~~~~~~~~~~~~

    def ropeend_rot(self, rot_a=np.pi/180, rot_axis=0):
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

    def ropeend_pos(
            self,
            pos_move=np.array([0., -1e-4, 0.]),
            bodymove_name="stiffrope"
        ):
        body_id = self.sim.model.body_name2id(bodymove_name)
        self.model.body_pos[body_id][:] += pos_move.copy()
        self.hold_pos(0.1)

    def ropeend_pos_all(
        self,
        pos_move=np.array([0., -1e-4, 0.]),
    ):
        body_id1 = self.sim.model.body_name2id(self.last_link_name)
        self.model.body_pos[body_id1][:] += pos_move
        body_id2 = self.sim.model.body_name2id("anchor_box")
        self.model.body_pos[body_id2][:] += pos_move
        body_id3 = self.sim.model.body_name2id("stiffrope")
        self.model.body_pos[body_id3][:] -= pos_move
        self.hold_pos(0.3)

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
        print("Giving force:")
        print(self.sim.data.xfrc_applied[body_id,:3])

    def sitename2pose(self, site_name):
        site_pose = np.zeros((4,4))
        site_id = self.sim.model.site_name2id(site_name)
        site_pose[:3,3] = self.sim.data.site_xpos[site_id]
        site_pose[:3,:3] = np.array(
            self.sim.data.site_xmat[site_id].reshape((3, 3)).copy()
        )
        site_pose[3,3] = 1.
        return site_pose
    
    def sitename2pos(self, site_name):
        site_id = self.sim.model.site_name2id(site_name)
        return self.sim.data.site_xpos[site_id]

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
        print("Giving velocity:")
        print(vel_dir)
        print(self.sim.data.body_xvelp[body_id])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| Rope things ||~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _update_ropepos(self):
        self.node_pos = self.der_sim.x.copy()

    def change_ropestiffness(self, alpha_bar, beta_bar):
        self.der_sim.change_ropestiffness(
            alpha_bar=alpha_bar,
            beta_bar=beta_bar,
        )

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| Pickle ||~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _save_initpickle(self):
        # input(self.env.observations['eef_pos'].copy())
        # original is rope moves 50, robot does not move
        mfs = self.max_fling_steps
        self.max_fling_steps = 9999999
        
        for i in range(50):
            self.ropeend_pos(np.array([-5e-3, 0., 0.]))
        start_qpos = self.observations['qpos'].copy()
        self.move_to_pos(
            targ_pos=np.array([-1.2, 0.6, 0.35]) # ([-1.2, 0.7, 0.35])
        )
        self.move_to_pos(
            targ_qpos=start_qpos
        )
        self.move_to_pos(
            targ_pos=self.robot_initpos[:3],
            targ_quat=T.axisangle2quat(self.robot_initpos[3:])    
        )
        # self.apply_force(force_dir=np.array([0., 15., 0.]))
        self.hold_pos(10.)
        # saving this sim state
        self.init_pickle = self.get_state()
        rl2_picklepath = 'rl2_' + str(self.r_pieces) + '.pickle' # 'rob3.pickle'
        rl2_picklepath = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data/" + rl2_picklepath
        )
        with open(rl2_picklepath, 'wb') as f:
            pickle.dump(self.init_pickle,f)
        self.max_fling_steps = mfs
        input('Pickle saved!')

    def _load_initpickle(self):
        rl2_picklepath = 'rl2_' + str(self.r_pieces) + '.pickle' # 'rob3.pickle'
        rl2_picklepath = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data/" + rl2_picklepath
        )
        with open(rl2_picklepath, 'rb') as f:
            self.init_pickle = pickle.load(f)
        self.set_state(self.init_pickle)
    
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
        self.zero_step()
        self._get_observations()
        self.der_sim._update_xvecs()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| Test things ||~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| Lhb things ||~~~~~~~~~~~~~~~~~~~~~~~~~~

    def lhb_init(self):
        # Twist 27 turns (27 * 360)
        s_ss_center = None
        s_ss_mincenter = 10.
        l_shorten = 0.3
        n_steps = 100
        step_len = l_shorten / n_steps / 2
        self.last_link_name = "r_link" + str(self.r_pieces) + "_body"
        # n_steps = int(l_shorten / step_len)
        # print(self.sitename2pos("r_joint0_site"))
        # print(self.sitename2pos("r_joint180_site"))
        # print(self.sitename2pos("r_joint0_site")-self.sitename2pos("r_joint180_site"))
        pos_move = np.array([step_len, -step_len, 0.])
        for i in range(2):
            self.ropeend_pos_all(pos_move=pos_move.copy())
        # pos_move = np.array([0., -step_len, 0.])
        # self.ropeend_pos_all(pos_move=-pos_move.copy())
        # self.ropeend_pos_all(pos_move=pos_move.copy())
        pos_move = np.array([step_len, step_len, 0.])
        for i in range(2):
            self.ropeend_pos_all(pos_move=pos_move.copy())
        
        pos_move = np.array([step_len, 0., 0.])
        for i in range(n_steps-2):
            # self.reset_vel()
            # print(self.sitename2pos("r_joint0_site"))
            # print(self.sitename2pos("r_joint180_site"))
            # print(self.sitename2pos("r_joint0_site")-self.sitename2pos("r_joint180_site"))
            self.ropeend_pos_all(pos_move=pos_move.copy())
            # self.ropeend_pos(pos_move=pos_move.copy())
            
            # print(i)
            if i % 20 == 0:
                self.reset_vel()
        # for i in range(27*360):
        #     self.ropeend_rot(rot_axis=0)
        # step_remain = l_shorten - 2* n_steps * step_len
        # self.ropeend_pos(np.array([step_remain, 0., 0.]))
        # print(self.sitename2pos("r_joint0_site"))
        # print(self.sitename2pos("r_joint0_site"))
        # print(self.sitename2pos("r_joint0_site")-self.sitename2pos("r_joint180_site"))
    
    def lhb_testing(self):
        for i in range(50):
            self.hold_pos(0.2)
            self.reset_vel()
        # self.reset_vel()
        # hold_time = 30.
        # init_time = self.cur_time
        # while (self.cur_time-init_time) < hold_time:
        #     self.step()

        s_ss, fphi = self.lhb_var_compute()
        fphi_min_id = np.where(fphi == fphi.min())[0][0]
        s_ss_min_id = floor(len(s_ss) / 2)
        s_ss_center = s_ss - s_ss[s_ss_min_id]
        fphi_center = fphi.copy()
        min_id_diff = fphi_min_id - s_ss_min_id
        if min_id_diff > 0:
            fphi_center[:-min_id_diff] = fphi[min_id_diff:]
            fphi_center[-min_id_diff:] = np.ones(min_id_diff)
        elif min_id_diff < 0:
            fphi_center[-min_id_diff:] = fphi[:min_id_diff]
            fphi_center[:-min_id_diff] = np.ones(-min_id_diff)

        # while (self.cur_time-init_time) < hold_time:
        #     self.step()
        #     s_ss, fphi = self.lhb_var_compute()
        #     maxdevi_id = np.where(fphi == fphi.min())[0][0]
        #     s_ss_maxdevi = abs(s_ss[maxdevi_id])
        #     if s_ss_maxdevi < s_ss_mincenter or s_ss_center is None:
        #         s_ss_mincenter = s_ss_maxdevi
        #         s_ss_center = s_ss.copy()
        #         fphi_center = fphi.copy()
        return s_ss_center, fphi_center
    
    def lhb_var_compute(self):
        joint_site_pos = np.array(
            self.sim.data.site_xpos[self.joint_site_idx[:]]
        ).copy()
        t_vec = joint_site_pos[1:] - joint_site_pos[:self.r_pieces]
        for i in range(len(t_vec)):
            t_vec[i] = t_vec[i] / np.linalg.norm(t_vec[i])
        e_x = np.array([-1., 0., 0.])
        devi_set = np.arccos(np.dot(t_vec, e_x))
        max_devi = np.max(devi_set)
        rot_step = self.der_sim.overall_rot / (self.r_pieces - 1)
        s_step = self.r_len / (self.r_pieces)
        s = np.zeros(len(devi_set))
        for i in range(len(devi_set)):
            s[i] = s_step * i + s_step/2
        s = s - np.mean(s)
        max_devi2 = max_devi
        # max_devi2 = 0.919
        s_ss = (
            s * (self.beta_bar*12/(2*self.alpha_bar))
            * np.sqrt((1-np.cos(max_devi2))/(1+np.cos(max_devi2)))
        )
        # s_ss = s_ss - np.mean(s_ss)
        fphi = (np.cos(devi_set)-np.cos(max_devi))/(1-np.cos(max_devi))
        return s_ss, fphi

    def start_lhbtest(self, new_start):
        # init vars
        self.avg_force = np.array([0., 0.])
        # Test for 1 rope type - length, alpha, beta
        # localized helical buckling test
        lhb_picklename = 'lhbtest{}.pickle'.format(self.r_pieces)
        lhb_picklename = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data/" + lhb_picklename
        )
        if new_start:
            self.lhb_init()
            self.init_pickle = self.get_state()
            with open(lhb_picklename, 'wb') as f:
                pickle.dump(self.init_pickle,f)
            print('Pickle saved!')
            # input('Pickle saved!')
        else:
            with open(lhb_picklename, 'rb') as f:
                self.init_pickle = pickle.load(f)

            # set overall rot
            self.init_pickle[0][9] = self.overall_rot
            self.p_thetan = self.overall_rot % (2. * np.pi)
            if self.p_thetan > np.pi:
                self.p_thetan -= 2 * np.pi
            self.init_pickle[0][10] = self.p_thetan
            self.set_state(self.init_pickle)

        s_ss_center, fphi_center =  self.lhb_testing()
            # s_ss_center = None
            # s_ss_mincenter = 10.
            # hold_time = 50.
            # init_time = self.cur_time
            # while (self.cur_time-init_time) < hold_time:
            #     self.step()
            #     # shouldn't this line onwards be out of the loop?
            #     # no, it is in the loop to ensure the final graphs are centralized
            #     s_ss, fphi = self.lhb_var_compute()
            #     maxdevi_id = np.where(fphi == fphi.min())[0][0]
            #     s_ss_maxdevi = abs(s_ss[maxdevi_id])
            #     if s_ss_maxdevi < s_ss_mincenter or s_ss_center is None:
            #         s_ss_mincenter = s_ss_maxdevi
            #         s_ss_center = s_ss.copy()
            #         fphi_center = fphi.copy()

        print(f"fphi = {fphi_center}")
        print(f"s_ss = {s_ss_center}")
        # print(f"max_devi = {max_devi}")
        pickledata = [fphi_center, s_ss_center]
        pickledata_path = 'lhb{}.pickle'.format(self.r_pieces)
        pickledata_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data/" + pickledata_path
        )
        with open(pickledata_path, 'wb') as f:
            pickle.dump(pickledata,f)
        print('pickled data')
        
        print(f"r_pieces = {self.r_pieces}")
        print(f"avg_force = {self.avg_force[0]}")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~|| Circle ||~~~~~~~~~~~~~~~~~~~~~~~~~~

    def circle_init(self):
        # assuming that rope axis is parallel to x axis
        self.stablestep = False
        y_offset = 0.3
        start_pos = self.bodyname2pos('stiffrope')
        end_pos = self.bodyname2pos(f'r_link{self.r_pieces}_body')
        e_pos = end_pos - start_pos
        step_len = 5e-3
        self.last_link_name = "r_link" + str(self.r_pieces) + "_body"
        n_steps = e_pos[0] / step_len
        if n_steps < 0:
            step_len = - step_len
            n_steps = int(-n_steps)
        step_ylen = 2 * y_offset / n_steps
        n_steps = int(n_steps/2)
        step_remain = e_pos[0] - 2* n_steps * step_len
        for i in range(n_steps):
            # input('stepped')
            # self.ropeend_pos(np.array([0., 5*step_ylen, 0.]), bodymove_name="anchor_box")
            # self.ropeend_pos(np.array([0., 5*step_ylen, 0.]), bodymove_name=self.last_link_name)
            # self.ropeend_pos_all(np.array([step_len, step_ylen, 0.]))
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

    def start_circletest(self, new_start):
        # Test for multiple rope types (vary alpha and beta bar)
        # alpha = 1.
        # r_len = 2*np.pi * self.r_pieces / (self.r_pieces-1)
        e_tol = 1000. # 0.075
        circtest_picklename = 'mbitest1.pickle'
        circtest_picklename = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data/" + circtest_picklename
        )
        if new_start:
            self.circle_oop = False
            self.circle_init()
            self.init_pickle = self.get_state()
            with open(circtest_picklename, 'wb') as f:
                pickle.dump(self.init_pickle,f)
            input(f'Circle Pickle saved!')
        else:
            with open(circtest_picklename, 'rb') as f:
                self.init_pickle = pickle.load(f)

            # set overall rot
            self.init_pickle[0][9] = self.overall_rot
            self.p_thetan = self.overall_rot % (2. * np.pi)
            if self.p_thetan > np.pi:
                self.p_thetan -= 2 * np.pi
            self.init_pickle[0][10] = self.p_thetan

            self.set_state(self.init_pickle)
            self.hold_pos(1.)

            self.apply_force(force_dir=np.array([0., 0., 1.]))
            self.hold_pos(0.5)
            self.apply_force(force_dir=np.array([0., 0., 0.]))
            # self.hold_pos(100.)
            # create a for loop that checks PCA error at each iter
            max_e = 0.
            for i in range(100):
                # self.ropeend_rot(rot_axis=0)
                # if not i % int(1.76*360) and i != 0:
                #     self.reset_vel()
                #     self.hold_pos(10.)
                if not i % 10:
                    self.reset_vel()
                #     self.hold_pos(0.3)
                # self.reset_vel()
                self.hold_pos(0.2)
                e_outofplane = self.check_e_PCA_circle()
                if e_outofplane > max_e:
                    max_e = e_outofplane
                # print(f'e_outofplane = {e_outofplane}')
            self.circle_oop = False
            print(f"e_tol = {e_tol}")
            print(f"e_outofplane = {e_outofplane}")
            print(f"max_e = {max_e}")
            if e_outofplane > e_tol or max_e > 5:
                self.circle_oop = True
                print(f'b_a = {self.beta_bar/self.alpha_bar} ==================================')
                print(f'out of plane theta_crit = {self.overall_rot} ==================================')
                # print(f'e_outofplane = {e_outofplane}')
            return e_outofplane
        

    # def str_mj_arr(arr):
    #     return ' '.join(['%0.3f' % arr[i] for i in range(arr._length_)])

    def print_collisions(self):
        d = self.sim.data
        print(f'No. of Collisions = {d.ncon}')
        for coni in range(d.ncon):
            print('  Contact %d:' % (coni,))
            con = d.obj.contact[coni]
            print('    dist     = %0.3f' % (con.dist,))
            print('    pos      = %s' % (self.str_mj_arr(con.pos),))
            print('    frame    = %s' % (self.str_mj_arr(con.frame),))
            print('    friction = %s' % (self.str_mj_arr(con.friction),))
            print('    dim      = %d' % (con.dim,))
            print('    geom1    = %d' % (con.geom1,))
            print('    geom2    = %d' % (con.geom2,))
        input()
