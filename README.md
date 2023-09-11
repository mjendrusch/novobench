# novobench

This package provides tools for benchmarking _de novo_ designed proteins
using AlphaFold2 and ESMfold following [1]. Given a set of designed protein
backbones and corresponding FASTA-format sequences, it predicts the structures
of each sequence and compares them to the designed backbone.

### dependencies
Before installing novobench, make sure you have the following installed:
* GCC or clang
* CUDA version 11.3 or later
in case your machine / cluster is set up with a module system, you will probably be able to get these using
```bash
module load GCC
module load CUDA
```

### installation
First, make sure `conda` is installed and create a conda environment:
```bash
conda create -n novobench python=3.9
conda activate novobench
```
Then, clone this repository and `cd` into it using
```bash
git clone https://github.com/mjendrusch/novobench.git
cd novobench
```
Finally, install dependencies using
```bash
source setup.sh
```

### usage
`novobench` runs structure predictor benchmarks on a set of protein backbones and corresponding sequences.
You provide `novobench` with a directory of backbone PDB-files and another directory of corresponding FASTA-files.
Each PDB-file (e.g. `backbone_0.pdb`) has to have a corresponding FASTA-file (`backbone_0.fa`) with the same base name.
`python -m novobench.run` then takes all FASTA-format sequences for a given backbone, predicts their structures
and compares these predictions with the original design.

You may choose different structure predictors for benchmarking. Currently, `novobench` supports AlphaFold2 [2] and ESMfold [3].
Additionally, `novobench` provides multiple options for structure prediction, allowing the use of structural templates,
as well as initialising structure prediction from a designed structure [1].

#### outputs
`novobench` will generate a directory of the following structure:
* `output-directory/`
  * `score.csv`: csv file containing benchmark statistics for each designed backbone and sequence
  * `predictions/`: directory containing predicted structures for each backbone and sequence
    * `backbone_0/`: directory containing predicted structures for each sequence for `backbone_0.pdb`
      * `design_0.pdb`: predicted structure of the first sequence in `backbone_0.fa`
      * ...
    * ...

Here, `score.csv` contains the values of relevant scores for quantifying the number of successful designs
and selecting designs for experimental validation. It contains the following columns:
* `result`: base name of the backbone PDB-file for this row, e.g. `backbone_0.pdb`
* `index`: index of the designed sequence for this row.
* `sequence`: designed sequence for this row.
* `sc_rmsd`: self-consistent RMSD between the designed backbone and the predicted structure of this row's sequence.
* `sc_tm`: self-consistent TM-score between designed and predicted structure.
* `plddt`: mean pLDDT over all non-templated positions in the predicted structure.
* `pae`: mean pAE of the predicted structure.
* `ipae`: mean interface pAE of the predicted structure, if multiple chains are present. Otherwise `inf`.
* `mpae`: minimum interface pAE of the predicted structure, if multiple chains are present. Otherwise `inf`.

#### recipe: _de novo_ design
* set `--model` to any AlphaFold model (`af_1` to `af_5`) or ESMfold (`esm`)
* set `--parameter-path` to the directory containing AlphaFold's parameters, or omit it, if using ESMfold.
* set `--prediction-mode` to `abinitio`.
* set `--pdb-path` to the directory containing your designed protein backbones.
* set `--fasta-path` to the directory containing your FASTA-format sequences for each backbone.
  In case your designed backbones contain sequence information, you may omit `--fasta-path`.
  `novobench` will then use the sequence in each PDB-file.
* set `--out-path` to the directory where `novobench` should write the output files.
  In case the provided path is not a directory, it will be created.

```bash
python -m novobench.run \
  --model "af_1" \
  --parameter-path $NOVOBENCH_PATH/alphafold/ \
  --prediction-mode "abinitio" \
  --pdb-path /path/to/backbone/pdbs/ \
  --fasta-path /path/to/desiged/fastas/ \
  --out-path /path/to/outputs/
```

#### recipe: binder design
* set `--model` to AlphaFold model 1 (`af_1`), as we need to handle structure templates for binder benchmarking
* set `--parameter-path` to the directory containing AlphaFold's parameters
* set `--prediction-mode` to `guess`. This initialises the recycled protein structure with the designed structure, following [1].
* set `--pdb-path` to the directory containing your designed protein backbones.
* set `--fasta-path` to the directory containing your FASTA-format sequences for each backbone.
  In case your designed backbones contain sequence information, you may omit `--fasta-path`.
  Novobench will then use the sequence in each PDB-file.
* set `--out-path` to the directory where `novobench` should write the output files.
  In case the provided path is not a directory, it will be created.
* set `--template` to the chain index of the target protein.
  E.g. 1 if your binder is chain A and your target protein is chain B.

```bash
# set the path to your novobench installation
export NOVOBENCH_PATH=/path/to/novobench
python -m novobench.run \
  --model "af_1" \
  --parameter-path $NOVOBENCH_PATH/alphafold/ \
  --prediction-mode "guess" \
  --pdb-path /path/to/pdb/backbone/directory \
  --fasta-path /path/to/fasta/sequence/directory \
  --out-path /path/to/output/directory \
  --templated chain-index-of-target-protein
```

#### parameters
* `--model`: structure predictor for benchmarking. Options:
  * AlphaFold (`af_1` to `af_5`)
  * ESMfold (`esm`)
* `--parameter-path`: path to the directory containing AlphaFold's parameters
* `--prediction-mode`: how to run prediction. Options:
  * `abinitio`: runs prediction from the provided sequence alone.
  * `guess`: runs prediction initialising recycling from the designed structure, following [1].
* `--templated`: list of chain indices to provide a template to AlphaFold for. Options:
  * `none`: no templates will be provided. This is the default setting.
  * `0,1,2`: comma-separated list of numerical chain indices. Listed chains will be used as a template for AlphaFold prediction.
* `--pdb-path`: path to directory containing PDB files.
* `--fasta-path`: path to directory containing FASTA files, or `none`, if designed PDB files contain all necessary sequence information.
  * NOTE: FASTA files need to have the same base name as their corresponding PDB file:
    e.g. "design_0.fa", if the corresponding PDB file is called "design_0.pdb"
* `--out-path`: path to output directory. If this is not a directory, it will be created. 

### References
[1] Nathaniel R. Bennett, Brian Coventry, Inna Goreshnik, Buwei Huang, Aza Allen, Dionne Vafeados, Ying Po Peng, Justas Dauparas, Minkyung Baek, Lance Stewart, Frank DiMaio, Steven De Munck, Savvas N. Savvides, and David Baker. Improving de novo protein binder design with deep learning. Nature Communications, 14, 2022.

[2] John M Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger, Kathryn Tunyasuvu-nakool, Russ Bates, Augustin Zídek, Anna Potapenko, Alex Bridgland, Clemens Meyer, Simon A A Kohl, Andy Ballard, Andrew Cowie, Bernardino Romera-Paredes, Stanislav Nikolov, Rishub Jain, Jonas Adler, Trevor Back, Stig Petersen, David A. Reiman, Ellen Clancy, Michal Zielinski, Martin Steinegger, Michalina Pacholska, Tamas Berghammer, Sebastian Bodenstein, David Silver, Oriol Vinyals, Andrew W. Senior, Koray Kavukcuoglu, Pushmeet Kohli, and Demis Hassabis. Highly accurate protein structure prediction with alphafold. Nature, 596:583 – 589, 2021.

[3] Rui Min Wu, Fan Ding, Rui Wang, Rui Shen, Xiwen Zhang, Shitong Luo, Chenpeng Su, Zuofan Wu, Qi Xie, Bonnie Berger, Jianzhu Ma, and Jian Peng. High-resolution de novo structure prediction from primary sequence. bioRxiv, 2022.