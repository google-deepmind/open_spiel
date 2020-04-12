#!/bin/bash

#SBATCH --job-name=egta_kuhn_poker_dqn
##SBATCH --job-name=egta_kuhn_poker_pg
##SBATCH --job-name=egta_kuhn_poker_ars
#SBATCH --mail-user=qmaai@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=5g
#SBATCH --time=02-00:00:00
#SBATCH --account=wellman1
#SBATCH --partition=standard
#SBATCH --output=/home/qmaai/se/open_spiel/open_spiel/python/algorithms/psro_v2/root_result/slurm_out.log
## The output is merely a place for slurm to ventilate its output. I don't want my home folder to be populated

module load python3.6-anaconda/5.2.0
cd ${SLURM_SUBMIT_DIR}
##python psro_v2_example.py --oracle_type=DQN --quiesce=False --gpsro_iterations=150 --number_training_episodes=10000 --sbatch_run=True
##python psro_v2_example.py --oracle_type=PG --quiesce=False --gpsro_iterations=150 --number_training_episodes=100000 --sbatch_run=True
##python psro_v2_example.py --oracle_type=ARS --quiesce=False --gpsro_iterations=150 --number_training_episodes=100000 --sbatch_run=True
python psro_v2_example.py --sbatch_run=True # default param from paul's 
