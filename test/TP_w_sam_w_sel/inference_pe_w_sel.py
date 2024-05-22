import numpy as np
import h5py
from intensity import pop_model_mcmc_samples_w_sel
import jax
from numpyro.infer import MCMC, NUTS

with h5py.File('pe_samples_w_selection.h5', 'r') as inp:
    ms_det = np.array(inp['ms_det'])
    zs_det = np.array(inp['zs_det'])
inp.close()

with h5py.File('Selection_samples.h5', 'r') as inp1:
    mdet = np.array(inp1['mdet'])
    zdet = np.array(inp1['zdet'])
inp1.close()

kernel = NUTS(pop_model_mcmc_samples_w_sel)

nmcmc = 1000
nchain = 1

mcmc = MCMC(kernel, num_warmup=nmcmc, num_samples=nmcmc, num_chains=nchain)
mcmc.run(jax.random.PRNGKey(np.random.randint(1<<32)), ms_det, zs_det, mdet, zdet)
samples = mcmc.get_samples()
with h5py.File('hyperparameter_pe_samples_w_sel.h5','w') as hf:
    for key,val in samples.items():
        hf.create_dataset(key,data = np.array(val))
hf.close()