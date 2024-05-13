import numpy as np
import h5py
from intensity import pop_model
import jax
from numpyro.infer import MCMC, NUTS

with h5py.File('mass_data.h5', 'r') as inp:
    mobs = np.array(inp['mobs'])
inp.close()

kernel = NUTS(pop_model)

nmcmc = 1000
nchain = 4

mcmc = MCMC(kernel, num_warmup=nmcmc, num_samples=nmcmc, num_chains=nchain)
mcmc.run(jax.random.PRNGKey(np.random.randint(1<<32)), mobs[mobs>50.0], sigma=5.0)
samples = mcmc.get_samples()
with h5py.File('samples.h5','w') as hf:
    for key,val in samples.items():
        hf.create_dataset(key,data = np.array(val))
hf.close()