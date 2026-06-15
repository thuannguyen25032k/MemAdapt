#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"
export EMBODIED_BENCH_ROOT=$(pwd)


# # Environment for ```Habitat and Alfred```
conda env create -f conda_envs/environment.yaml 
conda activate embench
pip install -e .

# Environment for ```EB-Navigation```
conda env create -f conda_envs/environment_eb-nav.yaml 
conda activate embench_nav
pip install -e .

# Environment for ```EB-Manipulation```
conda env create -f conda_envs/environment_eb-man.yaml 
conda activate embench_man
pip install -e .

# Install Git LFS
git lfs install
git lfs pull

# Install EB-ALFRED
conda activate embench
git clone git@hf.co:datasets/EmbodiedBench/EB-ALFRED
mv EB-ALFRED embodiedbench/envs/eb_alfred/data/json_2.1.0

# Install EB-Habitat
conda activate embench
conda install -y habitat-sim==0.3.0 withbullet  headless -c conda-forge -c aihabitat
git clone -b 'v0.3.0' --depth 1 https://github.com/facebookresearch/habitat-lab.git ./habitat-lab
cd ./habitat-lab
pip install -e habitat-lab
cd ..
conda install -y -c conda-forge git-lfs
python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets
mv data embodiedbench/envs/eb_habitat

# Install EB-Manipulation
conda activate embench_man
cd embodiedbench/envs/eb_manipulation
wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz
tar -xf CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz
rm CoppeliaSim_Pro_V4_1_0_Ubuntu20_04.tar.xz
mv CoppeliaSim_Pro_V4_1_0_Ubuntu20_04/ $EMBODIED_BENCH_ROOT
export COPPELIASIM_ROOT=$EMBODIED_BENCH_ROOT/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
# Persist CoppeliaSim env vars for future shell sessions
echo "export COPPELIASIM_ROOT=$EMBODIED_BENCH_ROOT/CoppeliaSim_Pro_V4_1_0_Ubuntu20_04" >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT' >> ~/.bashrc
echo 'export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT' >> ~/.bashrc
git clone https://github.com/stepjam/PyRep.git
cd PyRep
pip install -r requirements.txt
pip install -e .
cd ..
pip install -r requirements.txt
pip install -e .
cp ./simAddOnScript_PyRep.lua $COPPELIASIM_ROOT
git clone git@hf.co:datasets/EmbodiedBench/EB-Manipulation
mv EB-Manipulation/data/ ./
rm -rf EB-Manipulation/
cd ../../..

echo ""
echo "======================================================"
echo "Installation complete."
echo "NEXT STEPS before running any experiment:"
echo "  1. Source your shell config to pick up CoppeliaSim env vars:"
echo "       source ~/.bashrc   (bash)"
echo "       source ~/.zshrc    (zsh)"
echo "  2. Start the headless display server in a separate terminal:"
echo "       python -m embodiedbench.envs.eb_alfred.scripts.startx 1"
echo "  3. Verify each environment:"
echo "       conda activate embench     && python -m embodiedbench.envs.eb_alfred.EBAlfEnv"
echo "       conda activate embench     && python -m embodiedbench.envs.eb_habitat.EBHabEnv"
echo "       conda activate embench_nav && python -m embodiedbench.envs.eb_navigation.EBNavEnv"
echo "       conda activate embench_man && DISPLAY=:1 python -m embodiedbench.envs.eb_manipulation.EBManEnv"
echo "======================================================"
