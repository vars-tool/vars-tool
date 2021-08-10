import pandas as pd
import numpy as np

from tqdm.auto import tqdm
from time import sleep

from ..sa import gvars_funcs


# document this file
def star(parameters,
         num_stars,
         corr_mat,
         num_dir_samples,
         num_factors,
         report_verbose
         ):

    # load bar if report_verbose is true
    if report_verbose:
        stars_pbar = tqdm(desc='generating star points', total=10, dynamic_ncols=True)

    # Computing fictive correlation matrix
    # Note: that corr_mat and cov_mat are the same in terms of magnitude
    cov_mat = gvars_funcs.map_2_cornorm(parameters, corr_mat)
    if report_verbose:
        stars_pbar.update(1)

    # Generate independent standard normal samples
    # the amount of samples is the same as the amount of stars
    u = np.random.multivariate_normal(np.zeros(num_factors), np.eye(num_factors), num_stars)
    if report_verbose:
        stars_pbar.update(1)

    # Generate correlated standard normal samples
    # the amount of samples is the same as the amount of stars
    chol_u = np.linalg.cholesky(cov_mat)
    chol_u = chol_u.transpose()  # to get in correct format for matrix multiplication
    z = np.matmul(u, chol_u)  # transform samples to standard normal distribution
    if report_verbose:
        stars_pbar.update(1)

    # Generate Nstar actual multivariate samples x
    x = gvars_funcs.n2x_transform(z, parameters)
    if report_verbose:
        stars_pbar.update(1)

    # define index matrix of complement subset
    compsub = np.empty([num_factors, num_factors - 1])
    for i in range(0, num_factors):
        temp = np.arange(num_factors)
        compsub[i] = np.delete(temp, i)
    compsub = compsub.astype(int)
    if report_verbose:
        stars_pbar.update(1)

    # computer conditional variance and conditional expectation for each star center
    chol_cond_std = []
    std_cond_norm = []
    mui_on_noti = np.zeros((len(z), num_factors))
    for i in range(0, num_factors):
        noti = compsub[i]
        # 2 dimensional or greater matrix case
        if (cov_mat[noti, :][:, noti].ndim >= 2):
            cond_std = cov_mat[i][i] - np.matmul(cov_mat[i, noti],
                                                 np.matmul(np.linalg.inv(cov_mat[noti, :][:, noti]),
                                                           cov_mat[noti, i]))
            chol_cond_std.append(np.linalg.cholesky([[cond_std]]).flatten())
            std_cond_norm.append(cond_std)
            for j in range(0, len(z)):
                mui_on_noti[j][i] = np.matmul(cov_mat[i, noti],
                                              np.matmul(np.linalg.inv(cov_mat[noti, :][:, noti]), z[j, noti]))
        # less then 2 dimensional matrix case
        else:
            cond_std = cov_mat[i][i] - np.matmul(cov_mat[i, noti],
                                                 np.matmul(cov_mat[noti, :][:, noti], cov_mat[noti, i]))
            chol_cond_std.append(np.linalg.cholesky([[cond_std]]).flatten())
            std_cond_norm.append(cond_std)
            for j in range(0, len(z)):
                mui_on_noti[j][i] = np.matmul(cov_mat[i, noti], np.matmul(cov_mat[noti, :][:, noti] * z[j, noti]))
    if report_verbose:
        stars_pbar.update(1)

    # Generate directional sample:
    # Create samples in correlated standard normal space
    all_section_cond_z = []
    cond_z = []
    for j in range(0, num_dir_samples):
        stnrm_base = np.random.multivariate_normal(np.zeros(num_factors), np.eye(num_factors), num_stars)
        for i in range(0, num_factors):
            cond_z.append(stnrm_base[:, i] * chol_cond_std[i] + mui_on_noti[:, i])
        all_section_cond_z.append(cond_z.copy())
        cond_z.clear()
    if report_verbose:
        stars_pbar.update(1)

    # transform to original distribution and compute response surface
    Xi_on_Xnoti = []
    tmp1 = []
    Xi_on_Xnoti_and_Xnoti_temp = []
    Xi_on_Xnoti_and_Xnoti = []
    for j in range(0, num_dir_samples):
        for i in range(0, num_factors):
            tmp1.append(
                gvars_funcs.n2x_transform(np.array([all_section_cond_z[j][i]]).transpose(), parameters).flatten())
            tmp2 = x.copy()
            tmp2[:, i] = tmp1[i]
            Xi_on_Xnoti_and_Xnoti_temp.append(tmp2.copy())
            # attatch results from tmp1 onto Xi_on_Xnoti and Xi_on_Xnoti_and_Xnoti
        Xi_on_Xnoti.append(tmp1.copy())
        tmp1.clear()  # clear for next iteration
        Xi_on_Xnoti_and_Xnoti.append(Xi_on_Xnoti_and_Xnoti_temp.copy())
        Xi_on_Xnoti_and_Xnoti_temp.clear()  # clear for next iteration
    if report_verbose:
        stars_pbar.update(1)

    # Get star points into readable format
    params = [*(parameters)]
    star_points = {}
    points = {}
    temp = np.zeros([num_dir_samples, len(parameters)])
    for i in range(0, num_stars):
        for j in range(0, num_factors):
            for k in range(0, num_dir_samples):
                temp[k, :] = Xi_on_Xnoti_and_Xnoti[k][j][i]
            points[params[j]] = np.copy(temp)
        star_points[i] = points.copy()
    if report_verbose:
        stars_pbar.update(1)

    # put star points in a dataframe
    star_points_df = pd.concat(
        {key: pd.concat({k: pd.DataFrame(d) for k, d in value.items()}) for key, value in star_points.items()})
    star_points_df.index.names = ['centre', 'param', 'points']
    if report_verbose:
        sleep(0.1)
        stars_pbar.update(1)


    return star_points_df
