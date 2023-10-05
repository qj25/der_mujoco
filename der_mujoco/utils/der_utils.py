import numpy as np

def rot_vec(vec_0, u, a):
    # for rotation of 3d vector 
    # vec_0: vector
    # u: anchor vector (rotate about this vector)
    # a: angle in radians
    u = u / np.linalg.norm(u)
    R = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            if i == j:
                R[i,j] = (
                    np.cos(a)
                    + ((u[i]**2) * (1 - np.cos(a)))
                )
            else:
                ss = 1.
                if i < j:
                    ss *= -1
                if ((i+1)*(j+1)) % 2 != 0:
                    ss *= -1
                R[i,j] = (
                    u[i]*u[j] * (1 - np.cos(a))
                    + ss * u[3-(i+j)] * np.sin(a)
                )
    return np.dot(R,vec_0)

def ang_btwn(v1, v2):
    # vectors point away from the point where angle is taken
    # if np.linalg.norm(v1-v2) < 1e-6:
    #     return 0.
    cos_ab = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    if cos_ab > 1:
        return 0.
    ab = np.arccos(cos_ab)
    if np.isnan(ab):
        print(np.linalg.norm(v1))
        input(np.linalg.norm(v2))
    # if np.isnan(ab):
    #     print(np.dot(v1, v2)
    #     / (
    #         np.linalg.norm(v1)
    #         * np.linalg.norm(v2)
    #     ))
    #     print(np.dot(v1, v2))
    #     print(v1)
    #     print(np.linalg.norm(v1))
    #     print(v2)
    #     print(np.linalg.norm(v2))
    #     print(np.linalg.norm(v1)*np.linalg.norm(v2))
    #     print(ab)
    #     print('here')
    return ab

def ang_btwn2(v1, v2, v_anchor):
    # use sin and cos to find angle diff from -np.pi to np.pi
    # rotation angle of v1 to v2 wrt to axis v_anchor
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    v_anchor /= np.linalg.norm(v_anchor)
    e_tol = 1e-3
    if np.linalg.norm(v1-v2) < e_tol:
        return 0.
    sab = 0.
    cross12 = np.cross(v1, v2)
    # if np.linalg.norm(np.cross(cross12, v_anchor)) > e_tol:
    #     print(np.linalg.norm(np.cross(cross12, v_anchor)))
    #     print(np.linalg.norm(v1-v2))
    #     print('v1v2')
    #     print(v1)
    #     print(v2)
    #     print(cross12)
    #     print(v_anchor)
    #     raise Exception("inaccurate anchor")
    n_div = 0.
    for i in range(len(v1)):
        if abs(v_anchor[i]) < e_tol:
            continue
        sab += (
            cross12[i]
            / (
                np.linalg.norm(v1)
                * np.linalg.norm(v2)
            )
            / v_anchor[i]
        )
        n_div += 1.
    sab /= n_div
    cab = (
        np.dot(v1, v2)
        / (np.linalg.norm(v1) * np.linalg.norm(v2))
    )
    ab = np.arctan2(sab, cab)
    return ab

def ang_btwn3(v1, v2, v_anchor):
    theta_diff = np.arccos(v1.dot(v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
    if np.cross(v1,v2).dot(v_anchor) < 0:
        theta_diff *= -1.
    return theta_diff

def create_skewsym(v1):
    return np.array(
        [
            [0., -v1[2], v1[1]],
            [v1[2], 0., -v1[0]],
            [-v1[1], v1[0], 0.]
        ]
    )

if __name__ == "__main__":
    v1 = np.array([1.5, 1.5, 0.])
    v2 = np.array([-1., 0., 0.])
    va = np.array([0., 1., 0.])
    va = np.cross(v1,v2)
    va *= 1.

    print(ang_btwn2(v1,v2,va)/np.pi*180)
    print(ang_btwn3(v1,v2,va)/np.pi*180)