# Remaining work to be done for scPRINT-2

## Usability

- have a one line installer that manages the different environments people could
  use
  - [x] test on most GPUs
  - [x] test on CPUs
  - [ ] one line installer that also downloads the model, setup lamindb etc..
        (almost)
- have easy to use notebooks in the documentation website that explain simply
  and reproducibly the 13 different currently implemented capabilities of
  scprint2 by simplicity:
  - [ ] classification
  - [ ] classification+finetuning
  - [ ] denoising
  - [ ] batch correction
  - [ ] batch correction+finetuning
  - [ ] counterfactual reasoning
  - [ ] gene network inference
  - [ ] imputation on xenium
  - [ ] classification and embedding on xenium
  - [ ] cross species integration and classification and finetuning

## Open Questions

- what about only learning from contrastive loss (read the sccontrast paper
  first)
- apply the ECS at each class level embeddings !!!!!
- test fully quantized xpressor compressor (only FSQ, no VAEs) [1 day]

## Novel capabilities

- fine tuning class and notebook for classification from embeddings on
  additional labels (e.g. depmap /tcga diseases) [3 weeks]
- on new classes entirely (predict TCGA patient survival curves from cell
  embeddings too) [1 month]
- make it work on bulk RNAseq (fine-tuning on TCGA/GTEX/DepMap) [2 weeks]
- use 2nd GNN to work on multiple cells at once from a regular dataset and from
  spatial datasets [2 months]
- fine tuning on only very high quality / depth cells / long context (see if the
  model improves)

## Better model

- test MOE [3 days]
- predict kinetics too (add a second MLP on top of the gene embeddings) [1 week]
- Make a final V2 version with criss-cross attention and all other features
  (200M parameters and make it converge
- mix of bag-of-gene models and scFM
  [What about also predicting the gene name?? ](https://www.notion.so/What-about-also-predicting-the-gene-name-157f084143c3806ea47ef5afa9ff13f5?pvs=21)where
  we predict gene embedding with Wasserstein loss
- real diffusion denoising model (multiple iterations)
- test Deep Seek’s sparse attention / attention residual

## Perturbation space:

- explore the perturbation datasets
  - can we find perturbations that have no impact on cell state (might need to
    compare to average of perturbations)
  - can we find perturbations that have the same effect? (cell death, growth
    arrest, …)
- predict perturbations and pseudo perturbations based on temporal ordering
  - tahoe-100M chem-bert embedding + pseudo bulk version (group sets of 200
    cells and predict over
  - xaira and others with token + lora finetuning + pseudo bulk version
  - depmap / L1000 / PRISM survivability

## other random ideas

- explicit graph scPRINT.
  - need temporal training to maximize causality
  - use a graph transformer with learned base matrix
  - apply efficient retrieval of submatrix element like I currently have
  - use triplet-like attention with bias from alphafold model (open fold triton
    implementation)
  - apply loss
- temporal + spatial + perturbation modelling with secondary model
  - adopt a flow matching framework https://arxiv.org/pdf/2411.00698

# How to develop on this project

scPRINT-2 welcomes contributions from the community.

**You need PYTHON3!**

This instructions are for linux base systems. (Linux, MacOS, BSD, etc.)

## Setting up your own fork of this repo.

- On github interface click on `Fork` button.
- Clone your fork of this repo.
  `git clone git@github.com:YOUR_GIT_USERNAME/scPRINT-2.git`
- Enter the directory `cd scPRINT-2`
- Add upstream repo
  `git remote add upstream https://github.com/jkobject/scPRINT-2`

## Setting up your own virtual environment

Run `make virtualenv` to create a virtual environment. then activate it with
`source .venv/bin/activate`.

## Install the project in develop mode

Run `make install` to install the project in develop mode.

## Run the tests to ensure everything is working

Run `make test` to run the tests.

## Create a new branch to work on your contribution

Run `git checkout -b my_contribution`

## Make your changes

Edit the files using your preferred editor. (we recommend VIM or VSCode)

## Format the code

Run `make fmt` to format the code.

## Run the linter

Run `make lint` to run the linter.

## Test your changes

Run `make test` to run the tests.

Ensure code coverage report shows `100%` coverage, add tests to your PR.

## Build the docs locally

Run `make docs` to build the docs.

Ensure your new changes are documented.

## Commit your changes

This project uses
[conventional git commit messages](https://www.conventionalcommits.org/en/v1.0.0/).

Example: `fix(package): update setup.py arguments 🎉` (emojis are fine too)

## Push your changes to your fork

Run `git push origin my_contribution`

## Submit a pull request

On github interface, click on `Pull Request` button.

Wait CI to run and one of the developers will review your PR.

## Makefile utilities

This project comes with a `Makefile` that contains a number of useful utility.

```bash
❯ make
Usage: make <target>

Targets:
help:             ## Show the help.
install:          ## Install the project in dev mode.
fmt:              ## Format code using black & isort.
lint:             ## Run pep8, black, mypy linters.
test: lint        ## Run tests and generate coverage report.
watch:            ## Run tests on every change.
clean:            ## Clean unused files.
virtualenv:       ## Create a virtual environment.
release:          ## Create a new tag for release.
docs:             ## Build the documentation.
switch-to-poetry: ## Switch to poetry package manager.
init:             ## Initialize the project based on an application template.
```

## Making a new release

This project uses [semantic versioning](https://semver.org/) and tags releases
with `X.Y.Z` Every time a new tag is created and pushed to the remote repo,
github actions will automatically create a new release on github and trigger a
release on PyPI.

For this to work you need to setup a secret called `PIPY_API_TOKEN` on the
project settings>secrets, this token can be generated on
[pypi.org](https://pypi.org/account/).

To trigger a new release all you need to do is.

1. If you have changes to add to the repo
   - Make your changes following the steps described above.
   - Commit your changes following the
     [conventional git commit messages](https://www.conventionalcommits.org/en/v1.0.0/).
2. Run the tests to ensure everything is working.
3. Run `make release` to create a new tag and push it to the remote repo.

the `make release` will ask you the version number to create the tag, ex: type
`0.1.1` when you are asked.

> **CAUTION**: The make release will change local changelog files and commit all
> the unstaged changes you have.
