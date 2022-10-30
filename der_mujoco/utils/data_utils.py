import numpy as np

def compute_PCA(d1, d2, d3):
    n_data = len(d1)
    d123 = np.concatenate((
        [d1],
        [d2],
        [d3]
    ))


    d123_mean = np.mean(d123, axis=1, keepdims=True)
    svd = np.linalg.svd(d123 - np.mean(d123, axis=1, keepdims=True))
    left = svd[0]
    # print(left[:,-1])



    # print(d123)
    # d123_mean = np.zeros(3)
    # # normalizing and standardizing var
    # for i in range(3):
    #     d123_mean[i] = np.mean(d123[i])
    #     d123[i] -= d123_mean[i]
    #     if np.sum(np.square(d123[i])) != 0:
    #         d123[i] /= np.sqrt(np.sum(np.square(d123[i]))/(n_data-1))

    # covar_mat = np.dot(d123, np.transpose(d123)) / (n_data-1)
    # print(d123)
    # print(covar_mat)
    # eig_val, eig_vec = np.linalg.eig(covar_mat)
    # id_min = min(range(len(eig_val)), key=eig_val.__getitem__)
    # plane_norm = eig_vec[:,id_min]
    plane_norm = left[:,-1]
    plane_norm /= np.linalg.norm(plane_norm)
    plane_const = np.dot(plane_norm, d123_mean)
    # use formula for plane dist to calc dist norm of all points
    e_tot = 0.
    for i in range(n_data):
        e_tot += abs(np.dot(plane_norm, d123[:,i]) - plane_const)
        # print(abs(np.dot(plane_norm, d123[:,i]) - plane_const))
    # print(plane_norm)
    # plane_norm /= np.linalg.norm(plane_norm)
    # print(np.linalg.norm(plane_norm))
    return e_tot

if __name__ == "__main__":
    d1 = [0., 1., 2.]
    d2 = [0., 2., 0.]
    d3 = [0., 0., 1.]
    print(compute_PCA(d1, d2, d3))
    # t1 = (np.cross([1,2,0],[2,0,1]))
    # mag_1 = (np.linalg.norm(t1))
    # t1 = t1 / mag_1
    # print(t1)
    # print(np.linalg.norm(t1))