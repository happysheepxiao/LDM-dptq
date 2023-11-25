CUDA_VISIBLE_DEVICES=1 \
    python -u scripts/sample_bedroom.py \
    -r ./models/ldm/lsun_beds256/model.ckpt \
    -l out -n 5000 --batch_size 20 -c 200 \
    --n_bits_w 4 \
    --n_bits_a 6 \
    --channel_wise \
    --iters_w 20000 \
    --split \
    --sym \
    --cali_batch_size 25 \
    --recon_batch_size 5 \
    --train_batch_size 6 \
    # |& tee -i ./out/logs/2023-11-15-mission-1.txt 2>&1