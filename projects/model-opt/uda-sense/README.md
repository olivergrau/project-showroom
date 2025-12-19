# UdaSense: Optimized Object Recognition

In this project, you will develop a comprehensive compression pipeline for a pre-trained computer vision model designed to recognize household objects. Working as a Machine Learning Engineer at SmartHome Tech, you'll optimize their flagship app "udasense" to reduce model size, improve inference speed, and maintain accuracy for mobile deployment.

## Getting Started

These instructions will help you set up your development environment and understand the project structure.

### Installation

1. The project requires the following major dependencies:

```
pytorch>=1.12.0
torchvision>=0.13.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
```

Which are all collected within the `requirements.txt` file.

You can install for your environment by running `pip install -r requirements.txt`.


2. Install the project as a local package (this makes internal modules accessible). From the `starter-kit` directory run the following command:

```sh
pip install -e .
```


### Project Structure

```
├── notebooks/
│   ├── 01_baseline.ipynb - Establish baseline model performance
│   ├── 02_compression.ipynb - Implement & evaluate compression techniques
│   ├── 03_pipeline.ipynb - Design a multi-stage compression pipeline
│   └── 04_deployment.ipynb - Package model for mobile deployment
│
├── compression/
│   ├── __init__.py
│   ├── in-training/ - Compression techniques applied during training
│   │   ├── distillation.py
│   │   ├── gradual_pruning.py
│   │   └── quantization_aware.py
│   └── post-training/ - Techniques applied to trained models
│       ├── graph_optimization.py
│       ├── pruning.py
│       └── quantization.py
│
├── models/ - Where the baseline and optimized models will be saved
│
├── results/ - Where results for baseline and optimized models will be saved
│
├── utils/ - Helper modules
│   ├── __init__.py - Contains constants defining CTO targets
│   ├── compression.py - Compression utility functions
│   ├── data_loader.py - Dataset loading utilities
│   ├── evaluation.py - Model evaluation functions
│   ├── model.py - Model architecture definitions
│   └── visualization.py - Results visualization utilities
│
├── .gitignore 
├── .README 
├── requirements.txt - Packages required to run (and extend on) the project
├── report.md - (Template for) your final report
└── setup.py - Package setup file
```


## Project Instructions

Your task is to optimize a pre-trained computer vision model for mobile deployment while meeting these specific requirements:

- ✔️ Reduce model size by **30%**
- ✔️ Reduce inference time by **40%**
- ✔️ Maintain accuracy within **5%** of the baseline

**Within the `notebooks/` and `compression/` folders, you will find the TODOs for you to complete.**

> **IMPORTANT**: Always feel free to update any of the starter kit that's been provided to you if desired, even if outside of a *TODO*. This includes function definition, class definitions, variables, any other logic, and report template! 🤖

### Project Workflow

Your entry point for this project is the `notebooks/` folder.

1. Establish baseline performance ([`01_baseline.ipynb`](starter_kit/notebooks/01_baseline.ipynb))

    - Familiarize with code base for model training and evaluation
    - Review baseline model performance
    - Analyze model architecture and use case for compression opportunities


2. Implement and evaluate compression techniques ([`02_compression.ipynb`](starter_kit/notebooks/02_compression.ipynb))

    - Implement at least two different compression methods
    - Experiment with hyperparameters and configurations for each technique
    - Document the experimentation results and ideas for combining methods in the multi-stage compression pipeline

3. Design a multi-stage compression pipeline ([`03_pipeline.ipynb`](starter_kit/notebooks/03_pipeline.ipynb))

    - Define an implementation plan for the multi-stage compression pipeline
    - Combine techniques into an optimal compression strategy
    - Select the best performing pipeline based on requirements 
    - Report results and insights on different pipeline configurations

    > **NOTE**: You should try to meet the CTO requirements at this stage!

4. Package for mobile deployment ([`04_deployment.ipynb`](starter_kit/notebooks/04_deployment.ipynb))

    - Convert the optimized model to mobile-ready format
    - Verify functionality for mobile deployment
    - Collect insights and ideas for future work in a final analysis

5. Complete your final report  ([`report.md`](starter_kit/report.md))

    - Define an executive summary for your business audience
    - Collect the most important technical insights and results
    - Conclude with next steps and recommendations

### Deliverables

- [ ] Completed notebooks with all code cells executed
- [ ] Implementation of at least two compression techniques
- [ ] A multi-stage compression pipeline
- [ ] A mobile-ready optimized model
- [ ] A comprehensive report documenting your process and results

## Built With

* [PyTorch](https://pytorch.org/) - Deep learning framework
* [TorchVision](https://pytorch.org/vision/stable/index.html) - Computer vision tools and datasets
* [PyTorch Mobile](https://pytorch.org/mobile/home/) - Mobile deployment framework
