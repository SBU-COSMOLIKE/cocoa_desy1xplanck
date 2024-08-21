import sys
from os.path import join as pjoin
from mpi4py import MPI
import numpy as np
import torch
from cocoa_emu import Config, get_lhs_samples, get_params_list, CocoaModel, get_gaussian_samples
from cocoa_emu.emulator import NNEmulator, GPEmulator
from cocoa_emu.sampling import EmuSampler
import emcee

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

configfile = sys.argv[1]
config = Config(configfile)
label = config.emu_type.lower()
if hasattr(config, "gauss_temp"):
    label = label+f'_t{config.gauss_temp}'
if(rank==0):
    print("Initializing configuration space data vector dimension!")
    print("N_xip: %d"%(config.probe_size[0]//2))
    print("N_xim: %d"%(config.probe_size[0]//2))
    print("N_ggl: %d"%(config.probe_size[1]))
    print("N_w: %d"%(config.probe_size[2]))
    print("N_gk: %d"%(config.probe_size[3]))
    print("N_sk: %d"%(config.probe_size[4]))
    print("N_kk: %d"%(config.probe_size[5]))
    
n = int(sys.argv[2])

if(rank==0):
    print("Iteration: %d"%(n))
# ============== Retrieve training sample ======================
# Note that training sample does not include fast parameters
if(n==0):
    if(rank==0):
        if config.init_sample_type == "lhs":
            # retrieve LHS parameters
            # The parameter space boundary is set by config.lhs_mimax, which is 
            # the prior boundaries for flat prior and 4 sigma for Gaussian prior
            init_params = get_lhs_samples(config.n_dim, config.n_lhs, config.lhs_minmax)
        else:
            # retrieve Gaussian-approximation parameters
            # The mean of the Gaussian is specified by config.running_params_fid
            # plus shift from config.gauss_shift.
            init_params = get_gaussian_samples(config.running_params_fid, 
                config.running_params, config.params, config.n_resample, 
                config.gauss_cov, config.gauss_temp, config.gauss_shift)
    else:
        init_params = None
    init_params = comm.bcast(init_params, root=0)
    params_list = init_params
else:
    next_training_samples = np.load(pjoin(config.traindir, f'samples_{label}_{n}.npy'))
    params_list = get_params_list(next_training_samples, config.param_labels)
np.save(pjoin(config.traindir, f'total_samples_{label}_{n}.npy'), params_list)

# ================== Calculate data vectors ==========================

cocoa_model = CocoaModel(configfile, config.likelihood)

def get_local_data_vector_list(params_list, rank, return_s8=False):
    ''' Evaluate data vectors dispatched to the local process
    Input:
    ======
        - params_list: 
            full parameters to be evaluated. Parameters dispatched is a subset of the full parameters
        - rank: 
            the rank of the local process
    Outputs:
    ========
        - train_params: model parameters of the training sample
        - train_data_vectors: data vectors of the training sample
    '''
    train_params_list      = []
    train_data_vector_list = []
    train_sigma8_list      = []
    N_samples = len(params_list)
    N_local   = N_samples // size    
    for i in range(rank * N_local, (rank + 1) * N_local):
        if ((i-rank*N_local)%20==0):
            print(f'[{rank}/{size}] get_local_data_vector_list: iteration {i-rank*N_local}...')
        if type(params_list[i]) != dict:
            _p = {k:v for k,v in zip(config.running_params, params_list[i])}
        else:
            _p = params_list[i]
        params_arr  = np.array([_p[k] for k in config.running_params])
        # Here it calls cocoa to calculate data vectors at requested parameters
        data_vector, _s8 = cocoa_model.calculate_data_vector(_p, return_s8=return_s8)
        train_params_list.append(params_arr)
        train_data_vector_list.append(data_vector)
        if return_s8:
            train_sigma8_list.append(_s8)
    if return_s8:
        return train_params_list, train_data_vector_list, train_sigma8_list
    else:
        return train_params_list, train_data_vector_list, None

def get_data_vectors(params_list, comm, rank, return_s8=False):
    ''' Evaluate data vectors
    This function will further calls `get_local_data_vector_list` to dispatch jobs to and collect training data set from  other processes.
    Input:
    ======
        - params_list:
            Model parameters to be evaluated the model at
        - comm:
            MPI comm
        - rank:
            MPI rank
    Output:
    =======
        - train_params:
            model parameters of the training sample
        - train_data_vectors:
            data vectors of the training sample
    '''
    local_params_list, local_data_vector_list, local_sigma8_list = get_local_data_vector_list(params_list, rank, return_s8=return_s8)
    if rank!=0:
        comm.send([local_params_list, local_data_vector_list, local_sigma8_list], dest=0)
        train_params       = None
        train_data_vectors = None
        train_sigma8       = None
    else:
        data_vector_list = local_data_vector_list
        params_list      = local_params_list
        sigma8_list      = local_sigma8_list
        for source in range(1,size):
            new_params_list, new_data_vector_list, new_sigma8_list = comm.recv(source=source)
            data_vector_list = data_vector_list + new_data_vector_list
            params_list      = params_list + new_params_list
            sigma8_list      = sigma8_list + new_sigma8_list
        train_params       = np.vstack(params_list)    
        train_data_vectors = np.vstack(data_vector_list)
        train_sigma8       = np.vstack(sigma8_list)
    return train_params, train_data_vectors, train_sigma8

current_iter_samples, current_iter_data_vectors, current_iter_sigma8 = get_data_vectors(params_list, comm, rank, return_s8=True)
    
train_samples      = current_iter_samples
train_data_vectors = current_iter_data_vectors
train_sigma8       = current_iter_sigma8

# ============ Clean training data & save ====================
if(rank==0):
    # ================== Chi_sq cut ==========================
    def get_chi_sq_cut(train_data_vectors):
        chi_sq_list = []
        for dv in train_data_vectors:
            delta_dv = (dv - config.dv_lkl)[config.mask_lkl]
            chi_sq = delta_dv @ config.masked_inv_cov @ delta_dv
            chi_sq_list.append(chi_sq)
        chi_sq_arr = np.array(chi_sq_list)
        print(f'chi2 difference [{np.nanmin(chi_sq_arr)}, {np.nanmax(chi_sq_arr)}]')
        select_chi_sq = (chi_sq_arr < config.chi_sq_cut)
        return select_chi_sq
    # ===============================================
    select_chi_sq = get_chi_sq_cut(train_data_vectors)
    selected_obj = np.sum(select_chi_sq)
    total_obj    = len(train_data_vectors)
    print(f'[calculate_dv.py] Select {selected_obj} out of {total_obj}!')
    # ===============================================
        
    train_data_vectors = train_data_vectors[select_chi_sq]
    train_samples      = train_samples[select_chi_sq]
    train_sigma8       = train_sigma8[select_chi_sq]
    # ========================================================
    np.save(pjoin(config.traindir, f'data_vectors_{label}_{n}.npy'), train_data_vectors)
    np.save(pjoin(config.traindir, f'samples_{label}_{n}.npy'), train_samples)
    np.save(pjoin(config.traindir, f'sigma8_{label}_{n}.npy'), train_sigma8)
    # ======================================================== 
    print(f'Done data vector calculation iteration {n}!')
MPI.Finalize
