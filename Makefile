# compile Q1_nontiled.cu
Q1_nontiled: Q1_nontiled.cu
	nvcc -arch=sm_35 -O3 Q1_nontiled.cu -o Q1_nontiled

# run Q1_nontiled
run_nontiled: Q1_nontiled
	sbatch nontiled_batch.bash


# compile Q1_tiled.cu
Q1_tiled: Q1_tiled.cu
	nvcc -arch=sm_35 -O3 Q1_tiled.cu -o Q1_tiled

# run Q1_tiled
run_tiled: Q1_tiled
	sbatch tiled_batch.bash

# clean the directory
clean:
	rm -f Q1_nontiled Q1_tiled
