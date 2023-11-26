# Honours Project - Invexifying Regularisation of Non-Convex Machine Learning Models
This project implements all experiments and models for my Honours thesis, which focused on extending a novel 
regularisation technique introduced by [(R. Crane, F. Roosta)](https://arxiv.org/pdf/2111.11027v1.pdf), to consider 
more 
general convex 
loss functions and non-convex machine/deep 
learning architectures. We consider both deterministic and random initialisations, various datasets, as well as 
multiple optimisers.

For all random algorithms, a seed of 0 is used.

## Project Structure
- `datasets.py` - Handles all data pre-processing, subsetting, downloading, etc.
- `networks.py` - Defines all model architectures, including a wrapper class for the novel regularisation technique
- `workflow.py` - Implements all train/test/save procedures (generalised for each experiment).
- `hyperparameter_analysis.py` - Performs all plotting/analysis for hyperparameter tuning
- `regularisation_analysis.py` - Performs all plotting/analysis for comparing regularisation techniques
- `preliminary_plotting.py` - Handles plotting for preliminary experiments
- `Experiments/` - Contains scripts for all experiments.

All datasets downloaded by `datasets.py` are stored in a folder called `data`, that is created in the root directory.

At this stage, `workflow.py` is generalised to handle all model architectures/experiments for this project. Although,
it can easily be generalised further (e.g. via enums, function pointers, etc.).

### Dependencies

- `torchvision >= 0.15.2+cu118`
- `torch >= 2.0.1+cu118`
- `numpy >= 1.25.0`
- `matplotlib >= 3.7.1`
- `Pillow >= 9.3.0`

To ensure correct dependencies are met, one may run the following command:

> pip install -r requirements.txt