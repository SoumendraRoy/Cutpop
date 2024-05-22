import numpy as np
import h5py
from intensity import pop_model_mcmc_samples
import jax
from numpyro.infer import MCMC, NUTS

with h5py.File('pe_samples.h5', 'r') as inp:
    ms = np.array(inp['ms'])
    zs = np.array(inp['zs'])
inp.close()

kernel = NUTS(pop_model_mcmc_samples)

nmcmc = 1000
nchain = 1

mcmc = MCMC(kernel, num_warmup=nmcmc, num_samples=nmcmc, num_chains=nchain)
mcmc.run(jax.random.PRNGKey(np.random.randint(1<<32)), ms, zs)
samples = mcmc.get_samples()
with h5py.File('hyperparameter_pe_samples.h5','w') as hf:
    for key,val in samples.items():
        hf.create_dataset(key,data = np.array(val))
hf.close()