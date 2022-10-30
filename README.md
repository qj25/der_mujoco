# der_mujoco
Implementation of Discrete Elastic Rod ([Bergou2008](http://www.cs.columbia.edu/cg/pdfs/143-rods.pdf)) in [MuJoCo](https://mujoco.readthedocs.io/en/latest/overview.html).

How to use:
- simply create another MuJoCo environment using the derrope1d.py file as reference.

Note:
- model of the robot is available but not included in the environment.
- adjust timesteps to ensure stability
- adjust alpha_bar and beta_bar values to control bending and twisting stiffness of wire, respectively.
