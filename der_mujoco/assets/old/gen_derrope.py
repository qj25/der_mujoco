# changed: mass of geom

import os
import numpy as np
import der_muj.utils.transform_utils as T

class DERGenBase:
    def __init__(
        self,
        r_len=0.9,
        r_thickness=0.02,
        r_pieces=10,
        j_stiff=0.2,
        j_damp=0.7,
        # r_mass=None,
        init_pos=[0.53, 0.08, 0.13],
        init_quat=None,
        init_angle=0.,
        d_small=0.,
        both_fixed=True,
        rope_type="capsule",
        free_rope=False,
        vis_subcyl=False,
        obj_path=None,
    ):
        """
        connected by kinematic chain

        z-axis pointing from start section towards other sections.
        Displacement of other sections in the local positive z-direction.
        local axes:
        x - blue
        y - yellow
        params:
        - init_pos: starting position for base of rope/box
        - init_angle: rotation angle of the box about the z axis 
        """
        
        self.r_len = r_len
        self.r_thickness = r_thickness
        self.r_pieces = r_pieces
        self.j_stiff = j_stiff
        self.j_damp = j_damp
        # r_mass=None,
        self.init_pos = init_pos
        self.init_quat = init_quat
        self.init_angle = init_angle
        self.d_small = d_small
        self.both_fixed = both_fixed
        self.rope_type = rope_type
        self.free_rope = free_rope
        self.vis_subcyl = vis_subcyl
        self.obj_path = obj_path

        self._init_variables()
        self._write_mainbody()
        self._write_anchorbox()

    def _init_variables(self):
        self.max_pieces = self.r_len / self.r_thickness
        
        if self.r_pieces >= self.max_pieces:
            raise ValueError(
                'Too many sections for requested thickness.\n'
                + 'No.of must be strictly less than Max.\n'
                + f'No.of sections = {self.r_pieces}.\n'
                + f'Max sections = {self.max_pieces}.'
            )

        # check if rope is to be attached to box
        if self.init_quat is None:
            self.init_quat = np.array([0.5, 0.5, -0.5, 0.5])
            rot_axis = [0., 0., self.init_angle]
            rot_quat = T.axisangle2quat(rot_axis)
            self.init_quat = T.quat_multiply(rot_quat, self.init_quat)
        self.attach_pos = self.init_pos.copy()
        self.solref_val = "0.002 1"

        self.displace_link = self.r_len / self.r_pieces
        self.subcyl_len = 0.05
        self.box_size = [
                self.r_thickness, self.r_thickness,
                2*self.r_thickness,
            ]

        if self.obj_path is None:
            self.obj_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "derrope1d.xml"
            )
        self.obj_path2 = os.path.join(
            os.path.dirname(self.obj_path),
            "anchorbox.xml"
        )
        
        # write to file
        # tab character
        self.t = "  "
        self.cap_size = np.zeros(2)
        self.cap_size[0] = self.r_thickness / 2
        if self.rope_type == "capsule":
            self.cap_size[1] = (self.r_len / self.r_pieces - self.r_thickness) / 2
        elif self.rope_type == "cylinder":
            self.cap_size[1] = (self.r_len / self.r_pieces - self.d_small) / 2
        self.cap_size[1] -= self.d_small
        
        self.curr_tab = 0

    def _write_mb_geom(self, f, i_section):
        f.write(
            self.curr_tab*self.t
            + '<geom name="r_link{}_geom" type="{}" size="{:1.4f} {:1.4f}" density="0" rgba=".1 .2 .1 1" friction="1 0.005 0.0001" solref="{}" conaffinity="0" contype="0"/>\n'.format(
                i_section + 1,
                self.rope_type,
                self.cap_size[0],
                self.cap_size[1],
                self.solref_val,
            )
        )

    def _write_mainbody(self):
        with open(self.obj_path, "w+") as f:
            f.write('<mujoco model="stiff-rope">\n')
            self.curr_tab += 1
            # triangular prism mesh
            f.write(self.curr_tab*self.t + "<worldbody>\n")
            self.curr_tab += 1
            f.write(
                self.curr_tab*self.t 
                + '<body name="stiffrope" pos="{} {} {}" quat="1 0 0 0">\n'.format(
                    self.attach_pos[0],
                    self.attach_pos[1],
                    self.attach_pos[2],
                )
            )
            self.curr_tab += 1
            f.write(
                self.curr_tab*self.t
                + '<site name="ft_rope" pos="0 0 0" size="0.01 0.01 0.01" rgba="1 0 0 1" type="sphere" group="1" />\n'
            )
            # f.write(
            #     curr_tab*t 
            #     + '<body name="stiffrope" pos="{} {} {}" quat="{} {} {} {}">\n'.format(
            #         init_pos[0],
            #         init_pos[1],
            #         init_pos[2],
            #         init_quat[0],
            #         init_quat[1],
            #         init_quat[2],
            #         init_quat[3],
            #     )
            # )
            # curr_tab += 1
            f.write(
                self.curr_tab*self.t
                + '<body name="r_joint{}_body" pos="0 {} 0" quat="{} {} {} {}">\n'.format(
                    0,
                    self.displace_link / 2,
                    self.init_quat[0],
                    self.init_quat[1],
                    self.init_quat[2],
                    self.init_quat[3],
                )
            )
            if self.vis_subcyl:
                f.write(
                        (self.curr_tab+1)*self.t
                        + '<site name="subcyl_{}" pos="0 0 0" quat="0.707 0 0.707 0" size="0.001 {}" rgba="0 0 1 0.3" type="cylinder" group="1" />\n'.format(
                            0,
                            self.subcyl_len,
                        )
                    )
            f.write(
                (self.curr_tab+1)*self.t
                + '<site name="r_joint{}_site" type="sphere" size="0.0001 0.0001 0.0001"/>\n'.format(
                    0,
                )
            )
            f.write(self.curr_tab*self.t + '</body>\n')
            if self.free_rope:
                f.write(
                    self.curr_tab*self.t
                    + '<freejoint/>\n'
                )
            for i_section in range(self.r_pieces):
                if i_section == 0:
                    f.write(
                        self.curr_tab*self.t
                        + '<body name="r_link{}_body" pos="0 0 0" quat="{} {} {} {}">\n'.format(
                            i_section + 1,
                            self.init_quat[0],
                            self.init_quat[1],
                            self.init_quat[2],
                            self.init_quat[3],
                        )
                    )
                    self.curr_tab += 1
                else:
                    f.write(
                        self.curr_tab*self.t
                        + '<body name="r_link{}_body" pos="0 0 {}">\n'.format(
                            i_section + 1,
                            self.displace_link,
                        )
                    )
                    self.curr_tab += 1
                    f.write(
                        self.curr_tab*self.t
                        + '<joint name="r_joint{}" type="ball" pos="0 0 {}" stiffness="{}" damping="{}"/>\n'.format(
                            i_section + 1,
                            - self.displace_link/2,
                            self.j_stiff,
                            self.j_damp,
                        )
                    )
                f.write(
                    self.curr_tab*self.t
                    + '<body name="r_joint{}_body" pos="0 0 {}">\n'.format(
                        i_section + 1,
                        self.displace_link / 2,
                    )
                )
                if self.vis_subcyl:
                    f.write(
                        (self.curr_tab+1)*self.t
                        + '<site name="subcylx_{}" pos="0 0 0" quat="0.707 0 0.707 0" size="0.001 {}" rgba="0 0 1 0.3" type="cylinder" group="1" />\n'.format(
                            i_section + 1,
                            self.subcyl_len,
                        )
                    )
                f.write(
                    (self.curr_tab+1)*self.t
                    + '<site name="r_joint{}_site" type="sphere" size="0.0001 0.0001 0.0001"/>\n'.format(
                        i_section + 1,
                    )
                )
                f.write(self.curr_tab*self.t + '</body>\n')

                self._write_mb_geom(f, i_section)

                f.write(
                    self.curr_tab*self.t
                    + '<site name="r_link{}_site" type="sphere" size="0.0001 0.0001 0.0001"/>\n'.format(
                        i_section + 1,
                    )
                )
                if (
                    i_section == 0 
                    or i_section == self.r_pieces - 1
                    or i_section == np.floor(self.r_pieces / 2)
                ):
                    f.write(
                        self.curr_tab*self.t
                        + '<site name="twistcylx_{}" pos="0.05 0 0" quat="0.707 0 0.707 0" size="0.001 0.05" rgba="0 0 1 0.3" type="cylinder" group="1" />\n'.format(
                            i_section + 1,
                        )
                    )
                    f.write(
                        self.curr_tab*self.t
                        + '<site name="twistcyly_{}" pos="0 0.05 0" quat="0.707 -0.707 0 0" size="0.001 0.05" rgba="1 1 0 0.3" type="cylinder" group="1" />\n'.format(
                            i_section + 1,
                        )
                    )
                    
            self.curr_tab -= 1
            for i_section in range(self.r_pieces):
                f.write(self.curr_tab*self.t + '</body>\n')
                self.curr_tab -= 1
            f.write(2*self.t + '</body>\n')
            f.write(self.t + "</worldbody>\n")
            # f.write(t + '<sensor>\n')
            # f.write(2*t + '<force name="force_rope" site="ft_rope" />\n')
            # f.write(2*t + '<torque name="torque_rope" site="ft_rope" />\n')
            # f.write(t + '</sensor>\n')
            f.write('</mujoco>\n')

    def _write_equality(self, f):
        f.write(self.t + '<equality>\n')
        if self.both_fixed:
            f.write(
                self.t + "   <weld body1='anchor_box' body2='r_joint{}_body' solref='0.004 1' relpose='0 0 {} 0.5 0.5 -0.5 0.5'/>\n".format(
                    self.r_pieces,
                    self.box_size[1] # + self.r_len / self.r_pieces / 2,
                )
            )
        f.write(self.t + '</equality>\n')

    def _init_box_pos(self):
        len_shift = (
            self.r_len 
            - self.r_len / self.r_pieces / 2
            + self.box_size[1]
            + self.d_small
        )
        box_init_pos = self.init_pos.copy()
        box_init_pos[0] += len_shift * np.cos(self.init_angle-np.pi/2)
        box_init_pos[1] += len_shift * np.sin(self.init_angle-np.pi/2)
        return box_init_pos

    def _write_anchorbox(self):
        self.curr_tab = 0
        with open(self.obj_path2, "w+") as f:
            # create box
            f.write('<mujoco model="anchor-box">\n')
            self.curr_tab += 1
            # triangular prism mesh
            f.write(self.curr_tab*self.t + "<worldbody>\n")
            self.curr_tab += 1
            box_init_pos = self._init_box_pos()
            f.write(
                self.curr_tab*self.t + '<body name="anchor_box" pos="{} {} {}">\n'.format(
                    box_init_pos[0],
                    box_init_pos[1],
                    box_init_pos[2],
                    # init_quat[0],
                    # init_quat[1],
                    # init_quat[2],
                    # init_quat[3],
                )
            )
            self.curr_tab += 1
            f.write(
                self.curr_tab*self.t
                + '<geom name="fixedbox1" type="box" size="{:1.4f} {:1.4f} {:1.4f}" conaffinity="0" contype="0" rgba=".8 .2 .1 0.3" friction="1 0.005 0.0001"/>\n'.format(
                    self.box_size[0],
                    self.box_size[1],
                    self.box_size[2],
                )
            )
            self.curr_tab -= 1
            f.write(self.curr_tab*self.t + '</body>\n')
            self.curr_tab -= 1
            f.write(self.t + "</worldbody>\n")
            self._write_equality(f)
            # f.write(t + '<contact>\n')
            # curr_tab += 1
            # for i in range(1,r_pieces+1):
            #     for j in range(1,i):
            #         if abs(i-j) > 1:
            #             f.write(
            #                 curr_tab*t + "<pair geom1='r_link{}_geom' geom2='r_link{}_geom'/>\n".format(
            #                     i, j
            #                 )
            #             )
            # curr_tab -= 1
            # f.write(t + '</contact>\n')
            f.write('</mujoco>\n')

class DERGen_eq1D(DERGenBase):
    def __init__(
        self,
        r_len=0.9,
        r_thickness=0.02,
        r_pieces=10,
        j_stiff=0.2,
        j_damp=0.7,
        init_pos=[0.53, 0.08, 0.13],
        init_quat=None,
        init_angle=0,
        d_small=1e-5,
        both_fixed=True,
        rope_type="capsule",
        free_rope=False,
        vis_subcyl=False,
        obj_path=None
    ):
        super().__init__(
            r_len=r_len,
            r_thickness=r_thickness,
            r_pieces=r_pieces,
            j_stiff=j_stiff,
            j_damp=j_damp,
            init_pos=init_pos,
            init_quat=init_quat,
            init_angle=init_angle,
            d_small=d_small,
            both_fixed=both_fixed,
            rope_type=rope_type,
            free_rope=free_rope,
            vis_subcyl=vis_subcyl,
            obj_path=obj_path,
        )

        """
        connected by equality
        """
    
    def _write_mainbody(self):
        with open(self.obj_path, "w+") as f:
            f.write('<mujoco model="stiff-rope">\n')
            self.curr_tab += 1
            # triangular prism mesh
            f.write(self.curr_tab*self.t + "<worldbody>\n")
            self.curr_tab += 1
            f.write(
                self.curr_tab*self.t 
                + '<body name="stiffrope" pos="{} {} {}" quat="1 0 0 0">\n'.format(
                    self.attach_pos[0],
                    self.attach_pos[1],
                    self.attach_pos[2],
                )
            )
            self.curr_tab += 1
            f.write(
                self.curr_tab*self.t
                + '<site name="ft_rope" pos="0 0 0" size="0.01 0.01 0.01" rgba="1 0 0 1" type="sphere" group="1" />\n'
            )
            f.write(
                self.curr_tab*self.t
                + '<body name="r_joint{}_body" pos="0 {} 0" quat="{} {} {} {}">\n'.format(
                    0,
                    - self.displace_link / 2,
                    self.init_quat[0],
                    self.init_quat[1],
                    self.init_quat[2],
                    self.init_quat[3],
                )
            )
            if self.vis_subcyl:
                f.write(
                        (self.curr_tab+1)*self.t
                        + '<site name="subcyl_{}" pos="0 0 0" quat="0.707 0 0.707 0" size="0.001 {}" rgba="0 0 1 0.3" type="cylinder" group="1" />\n'.format(
                            0,
                            self.subcyl_len,
                        )
                    )
            f.write(
                (self.curr_tab+1)*self.t
                + '<site name="r_joint{}_site" type="sphere" size="0.0001 0.0001 0.0001"/>\n'.format(
                    0,
                )
            )
            f.write(
                (self.curr_tab+1)*self.t
                + '<geom name="r_joint{}_geom" type="sphere" size="0.01 0.01 0.01" mass="0.001" rgba=".8 .2 .1 1" friction="1 0.005 0.0001"/>\n'.format(
                    0,
                )
            )
            f.write(self.curr_tab*self.t + '</body>\n')
            # Loop here
            for i_section in range(self.r_pieces):
                if i_section > 0:
                    f.write(
                        self.curr_tab*self.t
                        + '<body name="r_link{}_body" pos="{} {} {}" quat="{} {} {} {}">\n'.format(
                            i_section + 1,
                            self.attach_pos[0],
                            self.attach_pos[1] + i_section * self.displace_link,
                            self.attach_pos[2],
                            self.init_quat[0],
                            self.init_quat[1],
                            self.init_quat[2],
                            self.init_quat[3],
                        )
                    )
                    self.curr_tab += 1
                    f.write(
                        self.curr_tab*self.t
                        + '<freejoint name="freejoint{}"/>\n'.format(i_section+1)
                    )
                else:
                    f.write(
                        self.curr_tab*self.t
                        + '<body name="r_link{}_body" pos="0 0 {}" quat="{} {} {} {}">\n'.format(
                            i_section + 1,
                            i_section * self.displace_link,
                            self.init_quat[0],
                            self.init_quat[1],
                            self.init_quat[2],
                            self.init_quat[3],
                        )
                    )
                    self.curr_tab += 1
                f.write(
                    self.curr_tab*self.t
                    + '<body name="r_joint{}_body" pos="0 0 {}">\n'.format(
                        i_section + 1,
                        - self.displace_link / 2,
                    )
                )
                if self.vis_subcyl:
                    f.write(
                        (self.curr_tab+1)*self.t
                        + '<site name="subcylx_{}" pos="0 0 0" quat="0.707 0 0.707 0" size="0.001 {}" rgba="0 0 1 0.3" type="cylinder" group="1" />\n'.format(
                            i_section + 1,
                            self.subcyl_len,
                        )
                    )
                f.write(
                    (self.curr_tab+1)*self.t
                    + '<site name="r_joint{}_site" type="sphere" size="0.0001 0.0001 0.0001"/>\n'.format(
                        i_section + 1,
                    )
                )
                f.write(
                    (self.curr_tab+1)*self.t
                    + '<geom name="r_joint{}_geom" type="sphere" size="0.01 0.01 0.01" mass="0.001" rgba=".8 .2 .1 1" friction="1 0.005 0.0001"/>\n'.format(
                        i_section + 1,
                    )
                )
                f.write(self.curr_tab*self.t + '</body>\n')

                self._write_mb_geom(f, i_section)

                f.write(
                    self.curr_tab*self.t
                    + '<site name="r_link{}_site" type="sphere" size="0.0001 0.0001 0.0001"/>\n'.format(
                        i_section + 1,
                    )
                )
                if (
                    i_section == 0 
                    or i_section == self.r_pieces - 1
                    or i_section == np.floor(self.r_pieces / 2)
                ):
                    f.write(
                        self.curr_tab*self.t
                        + '<site name="twistcylx_{}" pos="0.05 0 0" quat="0.707 0 0.707 0" size="0.001 0.05" rgba="0 0 1 0.3" type="cylinder" group="1" />\n'.format(
                            i_section + 1,
                        )
                    )
                    f.write(
                        self.curr_tab*self.t
                        + '<site name="twistcyly_{}" pos="0 0.05 0" quat="0.707 -0.707 0 0" size="0.001 0.05" rgba="1 1 0 0.3" type="cylinder" group="1" />\n'.format(
                            i_section + 1,
                        )
                    )
                self.curr_tab -= 1
                f.write(self.curr_tab*self.t + '</body>\n')
                if i_section == 0:
                    self.curr_tab -= 1
                    f.write(self.curr_tab*self.t + '</body>\n')
            f.write(self.t + "</worldbody>\n")
            # f.write(t + '<sensor>\n')
            # f.write(2*t + '<force name="force_rope" site="ft_rope" />\n')
            # f.write(2*t + '<torque name="torque_rope" site="ft_rope" />\n')
            # f.write(t + '</sensor>\n')
            f.write('</mujoco>\n')

    def _write_equality(self, f):
        f.write(self.t + '<equality>\n')
        if self.both_fixed:
            f.write(
                self.t + "   <weld body1='anchor_box' body2='r_joint{}_body' solref='0.004 1' relpose='0 {} 0 0.5 0.5 -0.5 0.5'/>\n".format(
                    self.r_pieces,
                    - self.box_size[1],
                )
            )
            # pass
            # f.write(
            #     self.t + "   <weld body1='anchor_box' body2='r_link{}_body' solref='0.002 1' relpose='0 {} 0 0.5 0.5 -0.5 0.5'/>\n".format(
            #         self.r_pieces,
            #         self.box_size[1] + self.r_len / self.r_pieces / 2,
            #     )
            # )
        for i_section in range(1,self.r_pieces):
            f.write(
                self.t + "   <connect body1='r_joint{}_body' body2='r_joint{}_body' active='true' solref='0.002 1' anchor='0 {} 0'/>\n".format(
                    i_section,
                    i_section + 1,
                    0.,
                )
            )
        # f.write(
        #     t + "   <connect body1='r_link{}_body' body2='table' anchor='−0.00000433 0.00097849 0.10596768' active='false' solref='0.02 1'/>\n".format(
        #         r_pieces,
        #     )
        # )
        f.write(self.t + '</equality>\n')

    def _init_box_pos(self):
        len_shift = (
            self.r_len 
            - self.r_len / self.r_pieces / 2
            + self.box_size[1]
            + self.d_small
        )
        box_init_pos = self.init_pos.copy()
        box_init_pos[0] += len_shift * np.cos(self.init_angle+np.pi/2)
        box_init_pos[1] += len_shift * np.sin(self.init_angle+np.pi/2)
        return box_init_pos

class DERGen_2D(DERGenBase):
    def __init__(
        self,
        r_len=0.9,
        r_thickness=0.02,
        r_pieces=10,
        j_stiff=0.2,
        j_damp=0.7,
        init_pos=[0.53, 0.08, 0.13],
        init_quat=None,
        init_angle=0,
        d_small=0.,
        both_fixed=True,
        rope_type="capsule",
        free_rope=False,
        vis_subcyl=False,
        obj_path=None
    ):
        super().__init__(
            r_len=r_len,
            r_thickness=r_thickness,
            r_pieces=r_pieces,
            j_stiff=j_stiff,
            j_damp=j_damp,
            init_pos=init_pos,
            init_quat=init_quat,
            init_angle=init_angle,
            d_small=d_small,
            both_fixed=both_fixed,
            rope_type=rope_type,
            free_rope=free_rope,
            vis_subcyl=vis_subcyl,
            obj_path=obj_path,
        )

    def _write_mb_geom(self, f, i_section):
        if i_section == 0 or i_section == self.r_pieces-1:
            f.write(
                self.curr_tab*self.t
                + '<geom name="r_link{}_geom" type="{}" size="{:1.4f} {:1.4f}" density="1000" rgba=".8 .2 .1 1" friction="1 0.005 0.0001" solref="{}" conaffinity="0" contype="0"/>\n'.format(
                    i_section + 1,
                    self.rope_type,
                    self.cap_size[0],
                    self.cap_size[1],
                    self.solref_val,
                )
            )
        else:
            f.write(
                self.curr_tab*self.t
                + '<geom name="r_link{}_geom" type="{}" size="{:1.4f} {:1.4f}" density="1000" rgba=".8 .2 .1 1" friction="1 0.005 0.0001" solref="{}"/>\n'.format(
                    i_section + 1,
                    self.rope_type,
                    self.cap_size[0],
                    self.cap_size[1],
                    self.solref_val,
                )
            )


def aa2quat(rot_axis, rot_angle):
    rot_quat = np.zeros(4)
    for i in range(3):
        rot_quat[i] = rot_axis[i] * np.sin(rot_angle/2)
    rot_quat[3] = np.cos(rot_angle/2)
    # input(rot_quat)
    return rot_quat

if __name__ == "__main__":
    DERGen_eq1D()
    """
    if fixed_box:
        box_size = [
            2*r_thickness,    # z size of box
            r_thickness, r_thickness
        ]
        first_j = -1
    if fixed_box:
        init_pos[2] += box_size[0]
        len_shift = (
            r_len 
            - r_len / r_pieces / 2
            + box_size[2]
        )
        init_pos[0] += - len_shift * np.cos(init_angle)
        init_pos[1] += - len_shift * np.sin(init_angle)
    if fixed_box:
            f.write(
                curr_tab*t
                + '<geom name="fixedbox1" type="box" size="{:1.4f} {:1.4f} {:1.4f}" rgba=".8 .2 .1 1" friction="1 0.005 0.0001"/>\n'.format(
                    box_size[0],
                    box_size[1],
                    box_size[2],
                )
            )
    """