# Raising the Bar in Graph-level Anomaly Detection (GLAD)

This is the companion code for a PyTorch implementation of graph-level anomaly detection methods described in the paper
**Raising the Bar in Graph-level Anomaly Detection** by Chen Qiu et al. 
The paper is published in IJCAI 2022 and can be found here https://arxiv.org/abs/2205.13845. 
The code allows the users to reproduce and extend the results reported in the study. Please cite the
above paper when reporting, reproducing or extending the results.

## Purpose of the project

This software is a research prototype, solely developed for and published as
part of the publication cited above. It will neither be maintained nor monitored in any way.

## Reproduce the Results

This repo contains the code of experiments with five methods (OCGTL,GTL,OCGIN,GTP,OCPool) on six graph datatsets.

Please run the command and replace \$# with available options (see below): 

```
python Launch_Exps.py --config-file $1 --dataset-name $2 
```

**config-file:** 

* config_OCGTL.yml; config_OCGIN.yml; config_GTL.yml; config_GTP.yml; config_OCPool.yml


**dataset-name:** 

* dd; thyroid; nci1; aids; imdb; reddit;

## How to Use
1. When using your own data, please put your data files under [DATA](DATA).

2. Create a config file which contains your hyper-parameters under [config_files](config_files).  

3. Add your data loader to the [loader/GraphDataClass.py](loader/GraphDataClass.py).


## Datasets

* Graph Data are downloaded from TUDataset https://chrsmrrs.github.io/datasets/. Please put the data under [DATA](DATA).  

## License

Raising the Bar in Graph-level Anomaly Detection (GLAD) is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in Raising the Bar in Graph-level Anomaly Detection (GLAD), see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).