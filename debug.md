# debug
```
cd /sfs/gpfs/tardis/home/mxk3rz/latent-diffusion

export PYTHONPATH=/sfs/gpfs/tardis/home/mxk3rz/latent-diffusion/src/taming-transformers:$PYTHONPATH

python - << 'PY'
import taming
print("taming imported from:", taming.__file__)
import taming.modules.vqvae.quantize as q
print("VectorQuantizer2:", q.VectorQuantizer2)
PY

```