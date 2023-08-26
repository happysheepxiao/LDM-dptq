CUDA_VISIBLE_DEVICES=2 \
    python -u scripts/sample_diffusion_qdrop_sym.py \
    -r models/ldm/lsun_beds256/model.ckpt \
    -l out -n 5000 --batch_size 20 -c 200 \
    --n_bits_w 4 \
    --n_bits_a 8 \
    --channel_wise \
    --iters_w 20000 \
    --split \
    |& tee -i ./out/logs/2023-08-26-mission-2.txt 2>&1
