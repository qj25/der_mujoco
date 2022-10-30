import matplotlib.pyplot as plt
import numpy as np
import gym
import random
from time import time
import der_mujoco.utils.transform_utils as T

from der_mujoco.utils.transform_utils import IDENTITY_QUATERNION

def check_proximity(v1, v2, check_mode='pos', d_tol=5e-4):
    # True if close, False if not close enough
    if check_mode == 'pos':
        distbtwn = v1 - v2
    elif check_mode == 'quat':
        distbtwn = T.quat_error(v1,v2)
        d_tol /= 200
        # input('checkquat')
    # print(np.linalg.norm(distbtwn[:]))
    if(
        np.linalg.norm(distbtwn[:]) < d_tol
    ):
        return True
    return False

class manip_rope_seq:
    def __init__(
        self,
        plot_dat=False,
    ):
        self.env = gym.make("der_mujoco:DerRope1D-v0")
        
        self.env.env.do_render = True

        self.env.reset()

        # settings
        # NIL

        # plot data
        self.plot_dat = plot_dat

        # intialize variables
        self.step_counter = 0
        self.eps_time = []
        self.move_step = []

    def plot_infodata(self):
        if not self.plot_dat:
            print("plot_dat set to: False.")
            return None
        # change to numpy array
        # self.move_step = np.array(self.move_step)

        # plotting
        # plt.figure("Movestep against Time")
        # plt.plot(self.eps_time, self.move_step[:,0])
        # plt.plot(self.eps_time, self.move_step[:,1])
        # plt.plot(self.eps_time, self.move_step[:,2])
        # plt.plot(self.eps_time, self.move_step[:,3])
        # plt.plot(self.eps_time, self.move_step[:,4])
        # plt.plot(self.eps_time, self.move_step[:,5])
        # plt.legend(
        #     ["x-pos", "y-pos", "z-pos", "x-rot", "y-rot", "z-rot"]
        # )
        # plt.ylabel('Movestep')
        # plt.xlabel('Time')
        # plt.grid()

        # plt.show()

mrs = manip_rope_seq(plot_dat=False)
# mrs.move_to_pos(targ_pos=start_pos)
# mrs.env.ropeend_pos(5e-4)
# mrs.env.hold_pos()
rot_angle = 720

# force_dir = np.array([0., 5., 0.])
# mrs.env.apply_force(body_name='anchor_box',force_dir=force_dir)

# vel_dir = np.array([0., 0.01, 0.])
# mrs.env.give_vel(body_name='anchor_box',vel_dir=vel_dir)
print('let')
for i in range(100):
    mrs.env.ropeend_pos(np.array([-5e-4, 5e-4, 0.]))
mrs.env.hold_pos(1000.)

# force_dir = np.array([0., 10., 0.])
# mrs.env.apply_force(body_name='anchor_box',force_dir=force_dir)

# input()
# start_t = time()
# for i in range(100):
#     mrs.env.ropeend_pos(np.array([20e-4, 5e-4, 0.]))

# end_t1 = time()
# print(f"overall1 = {end_t1 - start_t}")

# print('end pos move')
# mrs.env.hold_pos(2.)
# print('rotating in 1s')
# mrs.env.hold_pos(1.)

# for i in range(rot_angle):
#     mrs.env.ropeend_rot()
# print('end rot. moving next')

# mrs.env.hold_pos(2.)
# for i in range(100):
#     mrs.env.ropeend_pos(np.array([0, 5e-3, 0.]))
# mrs.env.hold_pos(10.)

# print(mrs.env.time_a / mrs.env.env_steps)

'1 {} 0 0.5 0.5 -0.5 0.5'

rot_angle *= 2

for i in range(135):
    mrs.env.ropeend_pos(np.array([-5e-3, -5e-3, 0.]))
print('rotating in 1s')
mrs.env.hold_pos(1.)
for i in range(rot_angle):
    mrs.env.ropeend_rot()
print('end rot.')
mrs.env.hold_pos(10.)
