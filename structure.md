Sẽ có 2 trainer: cho student trainer và teacher trainer
2 trainer sẽ có những hàm khá chung, tuy nhiên nhận dữ liệu và model sẽ khác nhau

sau khi train sẽ lưu hai ckpt
- Trainer checkpoint: Bao gồm thông tin optimizer, scheduler, hyperparams, model
- Model checkpoint: File checkpoint model only

Hyperparams của trainer:
- Max epoch: 
- experiment_path: lơi lưu các thử nghiệm, bao gồm trainer_ckpt và model_ckpt
- wandb_config: chứa những thông tin về wandb
- optimizer_config: chứa thông tin config optimizer
- scheduler_config: chứa thông tin config scheduler, chứa thêm một config nữa là on_step: True or on_epoch

trainer_ckpt:
trainer:
    optim:
        optimizer_state_dict:
        scheduler_state_dict:
    hyperparameters:
        ...