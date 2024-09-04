import sys, os
from os.path import join as pjoin
from mpi4py import MPI
import numpy as np
from cocoa_emu import Config, get_params_list, CocoaModel
os.environ["OMP_NUM_THREADS"] = "1"

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

configfile = sys.argv[1]
#eval_samples_fn = sys.argv[2]
#data_vector_fn = sys.argv[3]
label_valid = "gaussian_t64.0"
N_sample_valid = 100000
n = 0

config = Config(configfile)
### Load validation dataset
print(f'Loading validating data')
valid_samples = np.load(pjoin(config.traindir, 
    f'samples_{label_valid}_{N_sample_valid}_{n}.npy'))
valid_data_vectors = np.load(pjoin(config.traindir, 
    f'data_vectors_{label_valid}_{N_sample_valid}_{n}.npy'))
valid_sigma8 = np.load(pjoin(config.traindir, 
    f'sigma8_{label_valid}_{N_sample_valid}_{n}.npy'))

### Load emulators
probe_fmts = ['xi_p', 'xi_m', 'gammat', 'wtheta', 'gk', 'ks', 'kk']
probe_size = [config.probe_size[0]//2, 
              config.probe_size[0]//2, 
              config.probe_size[1], 
              config.probe_size[2], 
              config.probe_size[3], 
              config.probe_size[4], 
              config.probe_size[5]]
probe_params_mask = [config.probe_params_mask[0], 
                     config.probe_params_mask[0], 
                     config.probe_params_mask[1], 
                     config.probe_params_mask[2], 
                     config.probe_params_mask[3], 
                     config.probe_params_mask[4], 
                     config.probe_params_mask[5]]
emu_list = []
N_count = 0
for i,p in enumerate(probe_fmts):
    _l, _r = N_count, N_count + probe_size[i]
    fn = pjoin(config.modeldir, f'{p}_{n}_nn{config.nn_model}')
    if os.path.exists(fn+".h5"):
        print(f'Reading {p} NN emulator from {fn}.h5 ...')
        emu = NNEmulator(config.n_dim, probe_size[i], 
            config.dv_lkl[_l:_r], config.dv_std[_l:_r],
            config.inv_cov[_l:_r,_l:_r],
            mask=config.mask_lkl[_l:_r],param_mask=probe_params_mask[i],
            model=config.nn_model, device=device,
            deproj_PCA=True, lr=config.learning_rate, 
            reduce_lr=config.reduce_lr, 
            weight_decay=config.weight_decay, dtype="double")
        emu.load(fn)
    else:
        print(f'Can not find {p} emulator {fn}! Ignore probe {p}!')
        emu = None
    N_count += probe_size[i]
    emu_list.append(emu)
emu_sampler = EmuSampler(emu_list, config)

### Load sigma_8 emulator
fn = pjoin(config.modeldir, f'sigma8_{n}_nn{config.nn_model}')
if os.path.exists(fn+".h5"):
    print(f'Reading sigma8 NN emulator from {fn}.h5 ...')
    emu_s8 = NNEmulator(config.n_pars_cosmo, 1, config.sigma8_fid, 
            config.sigma8_std, 1.0/config.sigma8_std**2, 
            model=config.nn_model, device=device,
            deproj_PCA=False, lr=config.learning_rate, 
            reduce_lr=config.reduce_lr, 
            weight_decay=config.weight_decay, dtype="double")
    emu_s8.load(fn)
else:
    print(f'Can not find sigma8 emulator {fn}!')
    emu_s8 = None

### Compute dchi2
dchi2_list = []
dsigma8_list = []
assert valid_samples.shape[1]==config.n_dim, f'Inconsistent param dimension'+\
f'{valid_samples.shape[1]} v.s. {config.n_dim}'
for theta, dv, sigma8 in zip(valid_samples, valid_data_vectors, valid_sigma8):
    mv = emu_sampler.get_data_vector_emu(theta)
    diff = (dv-mv)[config.mask_lkl]
    dchi2 = diff@config.masked_inv_cov@diff
    dchi2_list.append(dchi2)
    sigma8_predict = emu_s8.predict(torch.Tensor(theta[:config.n_pars_cosmo]))[0]
    dsigma8_list.append(sigma8 - sigma8_predict)
dchi2_list = np.array(dchi2_list)
dsigma8_list = np.array(dsigma8_list)

frac_dchi2_1 = np.sum(dchi2_list>1.)/dchi2_list.shape[0]
frac_dchi2_2 = np.sum(dchi2_list>0.2)/dchi2_list.shape[0]
print(f'{frac_dchi2_1} chance of getting dchi2 > 1.0 from validation sample')
print(f'{frac_dchi2_2} chance of getting dchi2 > 0.2 from validation sample')

np.save(pjoin(config.traindir, "../dchi2_dsigma8_validation"),
	np.hstack([dchi2_list, dsigma8_list]))
