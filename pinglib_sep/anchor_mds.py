'''
    Majority of the code borrows from https://github.com/lab-robotics-unipv/mds_experiments
'''

import numpy as np

'''
    含锚点的多维标度降维
    输入：
        anchor_coords：锚点坐标，nx2
        tag_to_anchor_distances：m个待评估的点到锚点的距离，mxn
    返回：
        待评估点的坐标：mx2
'''


def anchored_mds_by_coords(anchor_coords, tag_to_anchor_distance, dim=2):
    anchor_amount = anchor_coords.shape[0]
    tag_amount = tag_to_anchor_distance.shape[0]

    from .anchored_mds.core import config, mds
    cfg = config.Config(no_of_anchors=anchor_amount, no_of_tags=tag_amount, noise=0, mu=0)
    cfg.set_anchors(anchor_coords)
    cfg.missingdata = True

    similarities = np.zeros((anchor_amount + tag_amount, anchor_amount + tag_amount))
    similarities[0:anchor_amount, 0:anchor_amount] = euclidean_distance(anchor_coords)
    similarities[0:anchor_amount, anchor_amount:] = np.transpose(tag_to_anchor_distance)
    similarities[anchor_amount:, 0:anchor_amount] = tag_to_anchor_distance

    coord_min = np.min(anchor_coords, axis=0)
    coord_max = np.max(anchor_coords, axis=0)
    init = np.zeros((tag_amount, dim))
    np.random.seed(0)
    for dim_id in range(dim):
        init[:, dim_id] = coord_min[dim_id] + \
                          np.random.random_sample(tag_amount) * (coord_max[dim_id] - coord_min[dim_id])
    init = np.concatenate([anchor_coords, init], axis=0)
    X, _, _, _ = mds._smacof_with_anchors_single(config=cfg, similarities=similarities,
                                                 metric=True, n_components=dim, init=init)
    return X[anchor_amount:, :]


#   未给定anchor坐标的情况下，首先通过MDS计算出anchor点的坐标
def anchored_mds_by_distance(anchor_to_anchor_distance, tag_to_anchor_distance=None, dim=2):
    from sklearn.manifold import MDS
    embedding = MDS(n_components=2, dissimilarity='precomputed')
    anchor_coords = embedding.fit(anchor_to_anchor_distance).embedding_

    anchor_amount = anchor_coords.shape[0]
    if tag_to_anchor_distance is None:
        tag_amount = 0
    else:
        tag_amount = tag_to_anchor_distance.shape[0]

    if tag_amount == 0:
        return anchor_coords, []

    from .anchored_mds.core import config, mds
    cfg = config.Config(no_of_anchors=anchor_amount, no_of_tags=tag_amount, noise=0, mu=0)
    cfg.set_anchors(anchor_coords)
    cfg.missingdata = True

    similarities = np.zeros((anchor_amount + tag_amount, anchor_amount + tag_amount))
    similarities[0:anchor_amount, 0:anchor_amount] = euclidean_distance(anchor_coords)
    similarities[0:anchor_amount, anchor_amount:] = np.transpose(tag_to_anchor_distance)
    similarities[anchor_amount:, 0:anchor_amount] = tag_to_anchor_distance

    coord_min = np.min(anchor_coords, axis=0)
    coord_max = np.max(anchor_coords, axis=0)
    init = np.zeros((tag_amount, dim))
    for dim_id in range(dim):
        init[:, dim_id] = coord_min[dim_id] + \
                          np.random.random_sample(tag_amount) * (coord_max[dim_id] - coord_min[dim_id])
    init = np.concatenate([anchor_coords, init], axis=0)
    X, _, _, _ = mds._smacof_with_anchors_single(config=cfg, similarities=similarities,
                                                 metric=True, n_components=dim, init=init)

    return X[0:anchor_amount, :], X[anchor_amount:, :]


#   计算一组点两两的距离
def euclidean_distance(coords):
    sample_amount = coords.shape[0]
    ones_vec = np.ones((sample_amount, 1))
    gram = np.matmul(coords, np.transpose(coords))
    gram_diag = np.expand_dims(np.diag(gram), axis=-1)
    edm = np.matmul(gram_diag, np.transpose(ones_vec)) + np.matmul(ones_vec, np.transpose(gram_diag)) - 2 * gram
    return np.sqrt(edm)
