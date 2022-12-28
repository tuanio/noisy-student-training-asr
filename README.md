conda create -n dev -c pytorch-nightly -c nvidia -c pytorch -c conda-forge python=3.9 pytorch torchaudio cudatoolkit pandas numpy 

pip install fairseq transformers torchsummary datasets evaluate torch-summary jiwer wandb matplotlib

wandb login 2cfd4b5c7e2828d412e5f871efea3a4c582efe18