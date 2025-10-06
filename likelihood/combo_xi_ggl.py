from cobaya.likelihoods.desy1xplanck._cosmolike_prototype_base import _cosmolike_prototype_base
import cosmolike_desy1xplanck_interface as ci
import numpy as np

class combo_xi_ggl(_cosmolike_prototype_base):
  def initialize(self):
    super(combo_xi_ggl,self).initialize(probe="xi_ggl")