import numpy as np
import der_mujoco.utils.transform_utils as T

MAX_PERLAYER = 10
MAX_LAYERS = 5

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

class PathNode:
    def __init__(
        self,
        pickle_data,
        parent,
        p_layer,
        pos_cmd, # pos_cmd taken to get to this node from its parent
        node_id
    ):
        self.pickle_data = pickle_data
        self.parent = parent
        self.node_id = node_id
        if parent is not None:
            self.layer = p_layer + 1
            self.pos_cmd = pos_cmd
        else:
            self.layer = 0
            self.action = None
        self.n_child = 0
        self.addable = True
        self.child = []

        # kd stuff
        self.kd_parent = None
        self.kd_child = np.zeros(2)

    def add_child(self, child_id):
        self.child.append(child_id)
        self.n_child += 1
        if self.n_child > MAX_PERLAYER-1:
            self.addable = False

    def del_child(self, child_id):
        self.child.remove(child_id)
        self.n_child -= 1
        self.addable = True

    def add_pose(self, curr_pos):
        self.curr_pos = curr_pos

    def add_kd_parent(self, kd_parent):
        self.kd_parent = kd_parent

    def add_kd_child(self, kd_child, which_child):
        # which child: 0 is less, 1 is more or equal to
        self.kd_child[which_child] = kd_child