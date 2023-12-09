1. First step is to install miniconda in a directory where you need to install all the relevant packages. Here the assumption is that directory where you need to install all the data is `/data/mn27889`

`mkdir -p /data/mn27889/miniconda3`
`wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /data/mn27889/miniconda3/miniconda.sh`
`bash /data/mn27889/miniconda3/miniconda.sh -b -u -p /data/mn27889/miniconda3`
`rm -rf /data/mn27889/miniconda3/miniconda.sh`

2. After installation, update the `.bashrc` and `.zshrc` with the correct binaries path:
`/data/mn27889/miniconda3/bin/conda init bash`
`/data/mn27889/miniconda3/bin/conda init zsh`

3. Now source both the `.bashrc` and `.zshrc` or open new terminals
`source ~/.bashrc`
`source ~/.zshrc`

4. `conda create -n vipergpt python=3.10`

5. `conda activate vipergpt`

6. Update $TORCH_HOME and $HF_HOME variable in `.bashrc` to make sure models are saved in the working directory:
`export TORCH_HOME=/data/mn27889/.cache/torch`
`export HF_HOME=/data/mn27889/.cache/huggingface`
`source ~/.bashrc`

7. Then install the PyTorch and CUDA libraries using:
`conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia`

8. Then install all the libraries listed in the `requirements.txt`:
`pip install -r requirements.txt`

