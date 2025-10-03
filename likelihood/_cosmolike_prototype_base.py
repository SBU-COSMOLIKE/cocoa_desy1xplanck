# Python 2/3 compatibility - must be first line
from __future__ import absolute_import, division, print_function
import os
import numpy as np
import scipy
import sys
import time

# Local
from cobaya.likelihoods.base_classes import DataSetLikelihood
from cobaya.log import LoggedError
from getdist import IniFile

from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline as _CubicSpline
import euclidemu2 as ee2
import math

import cosmolike_desy1xplanck_interface as ci

survey = "DES"

class _cosmolike_prototype_base(DataSetLikelihood):

  def initialize(self, probe):
    ini = IniFile(os.path.normpath(os.path.join(self.path, self.data_file)))
    self.probe = probe
    self.data_vector_file = ini.relativeFileName('data_file')
    self.cov_file = ini.relativeFileName('cov_file')
    self.mask_file = ini.relativeFileName('mask_file')
    self.lens_file = ini.relativeFileName('nz_lens_file')
    self.source_file = ini.relativeFileName('nz_source_file')
    self.lens_ntomo = ini.int("lens_ntomo")
    self.source_ntomo = ini.int("source_ntomo")
    self.ntheta = ini.int("n_theta")
    self.theta_min_arcmin = ini.float("theta_min_arcmin")
    self.theta_max_arcmin = ini.float("theta_max_arcmin")
    # CMB beam cut-off ------------------------------------------------
    self.lmax_kappa_cmb = ini.float("lmax_kappa_cmb")
    self.lmin_kappa_cmb = ini.float("lmin_kappa_cmb")
    self.fwhm  = ini.float("fwhm")
    self.pathHealpixWinFunc = ini.relativeFileName('pathHealpixWinFunc')
    self.is_cmb_bandpower = ini.int("is_cmb_bandpower", default = -1)
    if (1 == self.is_cmb_bandpower):
      self.binmat_file = ini.relativeFileName('binmat_file')
      self.offset_file = ini.relativeFileName('offset_file')
      self.nbp = ini.int("n_bp")
      self.lmin_bp = ini.float("lmin_bp")
      self.lmax_bp = ini.float("lmax_bp")
      self.ncl = 0
      self.lmin = 0
      self.lmax = 0
      self.is_cmb_kkkk_cov_from_sim = ini.int("is_cmb_kkkk_cov_from_sim",default = -1)
      if(self.is_cmb_kkkk_cov_from_sim == 1):
        Nvar = ini.float("Hartlap_Nvar")
        self.alpha_Hartlap = (Nvar - self.nbp -2.0)/(Nvar - 1.0) # < 1
      elif(self.is_cmb_kkkk_cov_from_sim == -1 or self.is_cmb_kkkk_cov_from_sim > 1):
        raise LoggedError(self.log,
            "MUST SPECIFY is_cmb_kkkk_cov_from_sim (0 or 1) IN THE DATA FILE!")
      else:
        self.is_cmb_kkkk_cov_from_sim = 0
        self.alpha_Hartlap = 1.0
    else:
      self.is_cmb_bandpower = 0
      self.ncl = ini.int("n_cl")
      self.lmin = ini.float("lmin")
      self.lmax = ini.float("lmax")
      self.binmat_file = 'none'
      self.offset_file = 'none'
      self.nbp = 0
      self.lmin_bp = 0
      self.lmax_bp = 0

    # ------------------------------------------------------------------------   
    tmp=int(1000 + 250*self.accuracyboost)
    self.z_interp_1D = np.concatenate((np.linspace(0.0,3.0,max(100,int(0.80*tmp))),
                                       np.linspace(3.0,50.1,max(100,int(0.40*tmp))),
                                       np.linspace(1070,1100,max(50,int(0.10*tmp)))),axis=0)
    self.len_z_interp_1D = len(self.z_interp_1D)

    tmp=int(min(120 + 20*self.accuracyboost,250))
    self.z_interp_2D = np.concatenate((np.linspace(0,3.0,max(50,int(0.75*tmp))), 
                                       np.linspace(3.01,50.0,max(30,int(0.25*tmp)))),axis=0)
    self.len_z_interp_2D = len(self.z_interp_2D)
    self.log10k_interp_2D = np.linspace(-4.99,2.0,int(1250+250*self.accuracyboost))
    self.len_log10k_interp_2D = len(self.log10k_interp_2D)
    # ------------------------------------------------------------------------

    ci.initial_setup()
    ci.init_probes(possible_probes=self.probe)
    ci.init_binning(self.ntheta, self.theta_min_arcmin, self.theta_max_arcmin)
    ci.init_cmb_bandpower(self.is_cmb_bandpower, 
                          self.is_cmb_kkkk_cov_from_sim, 
                          self.alpha_Hartlap)
    ci.init_cmb(self.lmin_kappa_cmb, 
                self.lmax_kappa_cmb, 
                self.fwhm, 
                self.pathHealpixWinFunc)
    if (1 == self.is_cmb_bandpower):
      ci.init_binning_cmb_bandpower(self.nbp, self.lmin_bp, self.lmax_bp)
    else:
      ci.init_binning_fourier(self.ncl, self.lmin, self.lmax)

    if self.use_emulator:
      ci.init_redshift_distributions_from_files(
          lens_multihisto_file=self.lens_file,
          lens_ntomo=int(self.lens_ntomo), 
          source_multihisto_file=self.source_file,
          source_ntomo=int(self.source_ntomo))
      ci.init_data(self.cov_file, self.mask_file, self.data_vector_file)  
      if (1 == self.is_cmb_bandpower):
        ci.init_cmb_bandpower_data(self.binmat_file, self.offset_file)
      ci.init_accuracy_boost(accuracy_boost=0.35, 
                             integration_accuracy=-1) # seems enough to compute PM
    else:
      ci.init_ntable_lmax(lmax=int(self.lmax))
      ci.init_accuracy_boost(accuracy_boost=self.accuracyboost, 
                             integration_accuracy=int(self.integration_accuracy))
      ci.init_cosmo_runmode(is_linear=False)

      if self.external_nz_modeling: 
        (self.lens_nz, self.source_nz) = ci.read_redshift_distributions(
            lens_multihisto_file=self.lens_file,
            lens_ntomo=int(self.lens_ntomo), 
            source_multihisto_file=self.source_file,
            source_ntomo=int(self.source_ntomo)) 
        ci.init_lens_sample_size(int(self.lens_ntomo))
        ci.init_source_sample_size(int(self.source_ntomo))
        ci.init_ntomo_powerspectra() # must be called after set_source/lens_size  
      else:
        ci.init_redshift_distributions_from_files(
          lens_multihisto_file=self.lens_file,
          lens_ntomo=int(self.lens_ntomo), 
          source_multihisto_file=self.source_file,
          source_ntomo=int(self.source_ntomo))

      ci.init_data(self.cov_file, self.mask_file, self.data_vector_file)
      if (1 == self.is_cmb_bandpower):
        ci.init_cmb_bandpower_data(self.binmat_file, self.offset_file)
      ci.init_IA(ia_model = int(self.IA_model), 
                 ia_redshift_evolution = int(self.IA_redshift_evolution))
      if self.probe != "xi":
        # (b1, b2, bs2, b3, bmag). 0 = one amplitude per bin
        ci.init_bias(bias_model=self.bias_model)

      if self.create_baryon_pca:
        self.use_baryon_pca = False
      if self.non_linear_emul == 1:
        self.emulator = ee2.PyEuclidEmulator()
      ci.init_baryons_contamination(
        self.use_baryonic_simulations_for_dv_contamination,
        self.which_baryonic_simulations_for_dv_contamination)

      # set up parameterization (CAMB v.s. CLASS)
      if self.boltzmann_code.lower() == "class":
        self.omega_m_str = "Omega_m"
        self.omega_b_str = "Omega_b"
        self.mnu_str = "m_ncdm_in_eV"
        self.w0_str = "w0_fld"
        self.wa_str = "wa_fld"
      elif self.boltzmann_code.lower() == "camb":
        self.omega_m_str = "omegam"
        self.omega_b_str = "omegab"
        self.mnu_str = "mnu"
        self.w0_str = "w"
        self.wa_str = "wa"
      else:
        raise ValueError(f'Can not recognize Boltzmann code {self.boltzmann_code}!')

    if self.use_baryon_pca:
      baryon_pca_file = ini.relativeFileName('baryon_pca_file')
      self.baryon_pcs = np.loadtxt(baryon_pca_file)
      ci.set_baryon_pcs(eigenvectors=self.baryon_pcs)
      self.log.info('use_baryon_pca = True')
      self.log.info('baryon_pca_file = %s loaded', baryon_pca_file)
      self.use_baryon_pca = True
      if self.subtract_mean:
        mean_baryon_diff_file = ini.relativeFileName('mean_baryon_diff_file')
        self.mean_baryon_diff = np.loadtxt(mean_baryon_diff_file)
        self.log.info('subtract_mean = True')
        self.log.info('mean_baryon_diff_file = %s loaded', mean_baryon_diff_file)
    else:
      self.log.info('use_baryon_pca = False')

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def get_requirements(self):
    if self.use_emulator:
      if self.probe == "xi":
        return {
          'cosmic_shear': None
        }
      elif self.probe == "3x2pt":
        return {
          "H0": None,
          'cosmic_shear': None,
          'ggl': None,
          'wtheta': None,
          'comoving_radial_distance': {
            "z": self.z_interp_1D 
          } # in Mpc
        }
      elif self.probe == "xi_gg":
        return {
          'cosmic_shear': None,
          'wtheta': None
        }
      elif self.probe == "xi_ggl":
        return {
          "H0": None,
          'cosmic_shear': None,
          'ggl': None,
          'comoving_radial_distance': {
            "z": self.z_interp_1D
          } # in Mpc
        }
      elif self.probe == "2x2pt":
        return {
          "H0": None,
          'ggl': None,
          'wtheta': None,
          'comoving_radial_distance': {
            "z": self.z_interp_1D 
          } # in Mpc
        }     
    else:
      res = {}
      if self.non_linear_emul == 1:
        res.update({"wa": None, "w": None, "mnu": None, "omegab": None})
      res.update({
          "As": None,
          "H0": None,
          "omegam": None,
          "Pk_interpolator": {
            "z": self.z_interp_2D,
            "k_max": self.kmax_boltzmann * self.accuracyboost,
            "nonlinear": (True,False),
            "vars_pairs": ([("delta_tot", "delta_tot")])
          },
          "comoving_radial_distance": {
            "z": self.z_interp_1D
          }, # in Mpc
          "Cl": { # DONT REMOVE THIS - SOME WEIRD BEHAVIOR IN CAMB WITHOUT WANTS_CL
            'tt': 0
          }
        })
      return res

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def set_cosmo_related(self):
    h = self.provider.get_param("H0")/100.0
    if not self.use_emulator:
      PKL  = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"), 
                                               nonlinear=False, 
                                               extrap_kmax=2.5e2*self.accuracyboost)
      lnPL = PKL.logP(self.z_interp_2D,
                      np.power(10.0,self.log10k_interp_2D)).flatten(order='F')+np.log(h**3)

      if self.non_linear_emul == 1:
        params = {
          'Omm'  : self.provider.get_param("omegam"),
          'As'   : self.provider.get_param("As"),
          'Omb'  : self.provider.get_param("omegab"),
          'ns'   : self.provider.get_param("ns"),
          'h'    : h,
          'mnu'  : self.provider.get_param("mnu"), 
          'w'    : self.provider.get_param("w"),
          'wa'   : self.provider.get_param("wa"),
        }
        # Euclid Emulator only works on z<10.0
        kbt, tmp_bt = ee2.get_boost2(params, 
                                     self.z_interp_2D[self.z_interp_2D < 10.0], 
                                     self.emulator, 
                                     10**np.linspace(-2.0589,0.973,self.len_log10k_interp_2D))
        bt = np.array(tmp_bt, dtype='float64')
        tmp = interp1d(np.log10(kbt), 
                       np.log(bt), 
                       axis=1,
                       kind='linear', 
                       fill_value='extrapolate', 
                       assume_sorted=True)(self.log10k_interp_2D-np.log10(h)) #h/Mpc
        tmp[:,10**(self.log10k_interp_2D-np.log10(h)) < 8.73e-3] = 0.0
        lnbt = np.zeros((self.len_z_interp_2D, self.len_log10k_interp_2D))
        lnbt[self.z_interp_2D < 10.0, :] = tmp
        # Use Halofit first that works on all redshifts
        lnPNL = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"),
          nonlinear=True,extrap_kmax=2.5e2*self.accuracyboost).logP(self.z_interp_2D,
          np.power(10.0,self.log10k_interp_2D)).flatten(order='F')+np.log(h**3) 
        # on z < 10.0, replace it with EE2
        lnPNL = np.where(
          (self.z_interp_2D<10)[:,None], 
          lnPL.reshape(self.len_z_interp_2D,self.len_log10k_interp_2D,order='F') + lnbt, 
          lnPNL.reshape(self.len_z_interp_2D,self.len_log10k_interp_2D,order='F')).ravel(order='F')
      elif self.non_linear_emul == 2:
        lnPNL = self.provider.get_Pk_interpolator(("delta_tot", "delta_tot"),
          nonlinear=True, extrap_kmax =2.5e2*self.accuracyboost).logP(self.z_interp_2D,
          np.power(10.0,self.log10k_interp_2D)).flatten(order='F')+np.log(h**3)   
      else:
        raise LoggedError(self.log, "non_linear_emul = %d is an invalid option", non_linear_emul)

      G_growth = np.sqrt(PKL.P(self.z_interp_2D,0.0005)/PKL.P(0,0.0005))*(1+self.z_interp_2D)
      G_growth /= G_growth[-1]

      ci.set_cosmology(
        omegam=self.provider.get_param("omegam"),
        H0=self.provider.get_param("H0"),
        log10k_2D=self.log10k_interp_2D-np.log10(h), #h/Mpc
        z_2D=self.z_interp_2D,
        lnP_linear=lnPL, 
        lnP_nonlinear=lnPNL, 
        G=G_growth,
        z_1D=self.z_interp_1D,
        chi=self.provider.get_comoving_radial_distance(self.z_interp_1D)*h # convert to Mpc/h
      )
    else:
      ci.set_distances(
        z=self.z_interp_1D,
        chi=self.provider.get_comoving_radial_distance(self.z_interp_1D)*h
      )

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def set_source_related(self, **params):
    ntomo = self.source_ntomo
    ci.set_nuisance_shear_calib(
      M=[params.get(p,0) for p in [survey+"_M"+str(i+1) for i in range(ntomo)]]
    )
    if not self.use_emulator:
      if self.external_nz_modeling: 
        # here we send n(z) at every point in the chain as the user may
        # modify it using an external function (example: adding outliers)
        # to modify it
        # (1) deep copy the numpy array (so we keep track of the fiducial
        # (2) modify the copy
        # (3) call set_source_sample
        source_nz_local = self.source_nz.copy()
        # insert mod function here <-
        #source_nz_local = f(source_nz_local, nuisance parameters)
        ci.set_source_sample(source_nz_local)
        # user may choose to still add photo-z bias or not (here we ad)
        ci.set_nuisance_shear_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_S"+str(i+1) for i in range(ntomo)]]
        )
      else:
        ci.set_nuisance_shear_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_S"+str(i+1) for i in range(ntomo)]]
        )
      ci.set_nuisance_ia(
        A1=[params.get(p,0) for p in [survey+"_A1_"+str(i+1) for i in range(ntomo)]],
        A2=[params.get(p,0) for p in [survey+"_A2_"+str(i+1) for i in range(ntomo)]],
        B_TA=[params.get(p,0) for p in [survey+"_BTA_"+str(i+1) for i in range(ntomo)]]
      )

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def set_lens_related(self, **params):
    ntomo = self.lens_ntomo
    ci.set_point_mass(
      PMV = [params.get(p, 0) for p in [survey+"_PM"+str(i+1) for i in range(ntomo)]]
    )
    if not self.use_emulator:
      ci.set_nuisance_bias(
        B1=[params.get(p,1) for p in [survey+"_B1_"+str(i+1) for i in range(ntomo)]],
        B2=[params.get(p,0) for p in [survey+"_B2_"+str(i+1) for i in range(ntomo)]],
        B_MAG=[params.get(p,0) for p in [survey+"_BMAG_"+str(i+1) for i in range(ntomo)]]
      )
      if self.external_nz_modeling: 
        # here we send n(z) at every point in the chain as the user may
        # modify it using an external function (example: adding outliers)
        # to modify it
        # (1) deep copy the numpy array (so we keep track of the fiducial
        # (2) modify the copy
        # (3) call set_source_sample
        lens_nz_local = self.lens_nz.copy()
        # insert mod function here <-
        #lens_nz_local = f(lens_nz_local, nuisance parameters)
        ci.set_lens_sample(lens_nz_local)
        # user may choose to still add photo-z bias or not (here we ad)
        ci.set_nuisance_clustering_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_L"+str(i+1) for i in range(ntomo)]],
          stretch=[params_values.get(p,0) for p in [survey+"_STRETCH_L"+str(i+1) for i in range(ntomo)]]
        )
      else:
        ci.set_nuisance_clustering_photoz(
          bias=[params.get(p,0) for p in [survey+"_DZ_L"+str(i+1) for i in range(ntomo)]],
          stretch=[params_values.get(p,0) for p in [survey+"_STRETCH_L"+str(i+1) for i in range(ntomo)]]
        )
  
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def compute_logp(self, datavector):
    return -0.5 * ci.compute_chi2(datavector)

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def logp(self, **params_values):
    datavector = self.internal_get_datavector(**params_values)
    return self.compute_logp(datavector)

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def get_datavector(self, **params):        
    if self.use_emulator:
      #dv = self.internal_get_datavector_emulator(**params)
      dv = 0.0
    else:
      dv = self.internal_get_datavector(**params)
    return np.array(dv,dtype='float64')

  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------
  # ------------------------------------------------------------------------

  def internal_get_datavector(self, **params):
    self.set_cosmo_related()
    if self.probe != "xi":
        self.set_lens_related(**params)
    self.set_source_related(**params)
    
    if self.create_baryon_pca:
      pcs = ci.compute_baryon_pcas(scenarios=self.baryon_pca_sims)
      np.savetxt(self.filename_baryon_pca, pcs)
    elif self.use_baryon_pca: 
      Q = [params.get(p,0) for p in [survey+"_BARYON_Q"+str(i+1) for i in range(self.npcs)]]     
      datavector = ci.compute_data_vector_masked_with_baryon_pcs(Q=Q)
    else:  
      datavector = ci.compute_data_vector_masked()

    if self.print_datavector:
      size = len(datavector)
      out = np.zeros(shape=(size, 2))
      out[:,0] = np.arange(0, size)
      out[:,1] = datavector
      fmt = '%d', '%1.8e'
      np.savetxt(self.print_datavector_file, out, fmt = fmt)
    return datavector
