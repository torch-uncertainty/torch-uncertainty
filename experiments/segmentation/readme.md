# Segmentation Benchmarks

Note: Optimize the number of `data.workers` to your computer to gain speed and avoid pauses.

## Experiments

To launch the segmentation experiments UNET trained and evaluated on MUAD, you can use the following command:

```bash
cd experiments/segmentation/muad
python main.py fit --config configs/muad/unet/{method}.yaml
```

`method` can be one of the following: `standard`, `deep_ensembles`, `mc_dropout`, `mimo`, `batch_ensemble`, `packed_ensembles`, and `bayesian`.

Note: feel free to change the `data.workers` parameter in the config file to optimize the number of workers for your machine, or set up multiple GPUs by changing the `trainer.devices` parameter.

You may find more configuration files in the [NeurIPS reproducibility repository](https://github.com/torch-uncertainty/torch-uncertainty-neurIPS-experiments).
