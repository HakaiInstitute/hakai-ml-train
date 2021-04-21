python ./lit_unet_kelp_presence.py \
/home/taylor/PycharmProjects/uav-classif/kelp/presence/train_input/data \
/home/taylor/PycharmProjects/uav-classif/kelp/presence/train_output/checkpoints \
--name=UNet --num_classes=2 \
--lr=0.01 --weight_decay=0.001 --gradient_clip_val=0.5 \
--auto_select_gpus --gpus=-1 --benchmark --sync_batchnorm \
--input_channels=3 --num_layers=5 --features_start=32 \
--max_epochs=100 --batch_size=2 --accumulate_grad_batches=4
# --overfit_batches=2  # TESTING
#  --max_epochs=100 --batch_size=8 --amp_level=O2 --precision=16 --distributed_backend=ddp --log_every_n_steps=10  # AWS