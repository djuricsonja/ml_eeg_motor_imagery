"""Riemannian recentering of covariance matrices."""

import numpy as np
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.base import invsqrtm


def recenter_to_identity(covs):
    """Whiten covariances by their Riemannian mean (mean -> identity)."""
    W = invsqrtm(mean_riemann(covs))
    return W @ covs @ W


def recenter_per_domain(covs, domains):
    """Recenter each domain (e.g. subject) to identity separately."""
    covs_rc = np.empty_like(covs)
    for d in np.unique(domains):
        idx = domains == d
        covs_rc[idx] = recenter_to_identity(covs[idx])
    return covs_rc
