import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
from os.path import join as pjoin
import numpy as np
import torch
from cocoa_emu import Config, get_lhs_params_list, get_params_list, CocoaModel
from cocoa_emu.emulator import NNEmulator, GPEmulator
from cocoa_emu.sampling import EmuSampler
import emcee
from argparse import ArgumentParser

### Parallelization
#from multiprocessing import Pool
from schwimmbad import MPIPool

### This file use pre-trained emulator to run MCMC chains based on input YAML
### Usage: ${PYTHON3} sample_emulator.py ${CONFIG}
parser = ArgumentParser()
parser.add_argument('config', type=str, help='Configuration file')
args = parser.parse_args()

#if torch.cuda.is_available():
#    device = torch.device('cuda')
#    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
#else:
#    device = torch.device('cpu')
#    torch.set_num_interop_threads(40) # Inter-op parallelism
#    torch.set_num_threads(40) # Intra-op parallelism
#torch.set_default_device(device)
device=torch.device("cpu")
print('Using device: ', device)

if __name__ == '__main__':
	configfile = args.config
	config = Config(configfile)
	assert config.emu_type.lower()=='nn', f'Only support NN emulator now!'
	
	### read emulators
	probe_fmts = ['xi_pm', 'gammat', 'wtheta', 'gk', 'ks', 'kk']
	emu_list = []
	for i,p in enumerate(probe_fmts):
		_l, _r = sum(config.probe_size[:i]), sum(config.probe_size[:i+1])
		fn = pjoin(config.modeldir, f'{p}_nn{config.nn_model}')
		if os.path.exists(fn+".h5"):
			print(f'Reading {p} NN emulator from {fn}.h5 ...')
			emu = NNEmulator(config.n_dim, config.probe_size[i], 
				config.dv_lkl[_l:_r], config.dv_std[_l:_r],
				config.inv_cov[_l:_r,_l:_r],
				mask=config.mask_lkl[_l:_r],
				param_mask=config.probe_params_mask[i],
				model=config.nn_model, device=device,
				deproj_PCA=True, lr=config.learning_rate, 
    			reduce_lr=config.reduce_lr, 
    			weight_decay=config.weight_decay, dtype="double")
			emu.load(fn)
		else:
			print(f'Can not find {p} emulator {fn}! Ignore probe {p}!')
			emu = None
		emu_list.append(emu)
	# read sigma8 emulator
	fn = pjoin(config.modeldir, f'sigma8_nn{config.nn_model}')
	if os.path.exists(fn+".h5"):
		print(f'Reading sigma8 NN emulator from {fn}.h5 ...')
		emu_s8 = NNEmulator(config.n_pars_cosmo, 1, config.sigma8_fid, 
				config.sigma8_std, np.atleast_2d(1.0/config.sigma8_std**2), 
    			model=config.nn_model, device=device,
    			deproj_PCA=False, lr=config.learning_rate, 
    			reduce_lr=config.reduce_lr, 
    			weight_decay=config.weight_decay, dtype="double")
		emu_s8.load(fn)
	else:
		print(f'Can not find sigma8 emulator {fn}!')
		emu_s8 = None

	### Build EmuSampler
	emu_sampler = EmuSampler(emu_list, config)
	pos0 = emu_sampler.get_starting_pos()

	def ln_prob_wrapper(theta, temper=1.0):
		return emu_sampler.ln_prob(theta, temper)

	### Run emcee
	with MPIPool() as pool:
		if not pool.is_master():
			pool.wait()
			sys.exit(0)
		sampler = emcee.EnsembleSampler(config.n_emcee_walkers, emu_sampler.n_sample_dims, ln_prob_wrapper, pool=pool)
		sampler.run_mcmc(pos0, config.n_mcmc, progress=True)

	samples = sampler.get_chain(discard=config.n_burn_in, thin=config.n_thin, flat=True)
	logprobs= sampler.get_log_prob(discard=config.n_burn_in, thin=config.n_thin, flat=True)

	### Get sigma_8 for the chain
	if emu_s8 is not None:
		print(f'Getting sigma8 for the emcee chain...')
		derived_sigma8 = emu_s8.predict(torch.Tensor(samples[:,:config.n_pars_cosmo]))
		np.save(pjoin(config.chaindir, config.chainname+'.npy'), 
				np.hstack([samples, derived_sigma8, logprobs[:,np.newaxis]]))
	else:
		np.save(pjoin(config.chaindir, config.chainname+'.npy'), 
			    np.hstack([samples, logprobs[:,np.newaxis]]))
	print("Done!")
