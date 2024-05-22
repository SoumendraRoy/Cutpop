import numpy as np
import h5py
from intensity import pop_model_true
import jax
from numpyro.infer import MCMC, NUTS

with h5py.File('truth.h5', 'r') as inp:
    mtrue = np.array(inp['mtrue'])
    ztrue = np.array(inp['ztrue'])
inp.close()

kernel = NUTS(pop_model_true)

nmcmc = 1000
nchain = 1

mcmc = MCMC(kernel, num_warmup=nmcmc, num_samples=nmcmc, num_chains=nchain)
mcmc.run(jax.random.PRNGKey(np.random.randint(1<<32)), mtrue, ztrue)
samples = mcmc.get_samples()
with h5py.File('hyperparameter_true.h5','w') as hf:
    for key,val in samples.items():
        hf.create_dataset(key,data = np.array(val))
hf.close()