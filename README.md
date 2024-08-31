## Fiber-Reinforced Constitutive Neural Network for Hyperelastic Biological Tissues: Incorporating Physical Prior Knowledge
This repository contains the code for our paper "Fiber-Reinforced Constitutive Neural Network for Hyperelastic Biological Tissues: Incorporating Physical Prior Knowledge".
The FHCNNBio framework enables rapid modeling of material strain-stress relationships from data, while also allowing for parametrization of the underlying constitutive model to fine-tune the mechanical behavior of materials for specific application scenarios.
> It is important to note that, in order to meet the specific demands of individual experiments, our code architecture may undergo targeted refinements to accommodate distinct experimental conditions. Consequently, certain code segments may operate correctly under particular circumstances but fail to execute under others. These are minor limitations, and we will maintain an ongoing commitment to updating the repository to resolve these issues.
Furthermore, there are some experiments that are relatively complex and do not share the same configuration environment; we require some time to organize the code, and we will update it soon. --2024.8.31

## Dataset
Our training dataset is composed mainly of real-world data extracted from existing published literature, and the compilation of these datasets is available in repository https://github.com/CPShub/sim-data. In addition, the data used for finite element training and validation were generated using the [FEBio](https://febio.org) software. The relevant generation codes and material settings can be accessed in the folder **Data/fem**. Note that prior installation of Febio is required.

## Usage
The code is primarily built using the latest PyTorch implementation; therefore, **in general, once you have installed the latest PyTorch and its corresponding Python dependencies, you can utilize your preferred editor to read and run the code** (PyCharm is recommended).
- **Data**
  The comprehensive dataset employed for training and validation purposes is provided, including the Mathematica notebooks utilized for analytical verification.
- **benchmark**
  Benchmark testing code is provided to validate the correctness of our rewritten PyTorch implementation from the original Jax code.
- **comutils**
- **experiments**
  The complete collection of experimental code is provided, incorporating finite element simulation experiments.
- **logs**
- **outputs**
- **src**
  Source code for our methodology and model implementations, including computational realizations of certain continuum mechanics theories.
## Citation
Our manuscript is presently undergoing rigorous evaluation through the peer-review process.
## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
