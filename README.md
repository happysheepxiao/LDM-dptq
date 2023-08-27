# LDM-qdiff

## Requirements
For generation, a suitable running conda environment named `ldm` can be created
and activated with:

```
conda env create -f ldm.yaml
conda activate ldm
```

For FID calculation, A suitable running conda environment named `fid` can be created
and activated with:

```
conda env create -f fid.yaml
conda activate fid
```

Please download the model file for quantization at https://ommer-lab.com/files/latent-diffusion/lsun_bedrooms.zip

Please download the reference batch for FID calculation at https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/lsun/bedroom/VIRTUAL_lsun_bedroom256.npz

## Generation
For LSUN-bedrooms，run the script via
```
bash run_qdrop.sh
```
