# Distributed Certifiably Correct Range-Aided SLAM (DCORA)

## Installation

Install the required system dependencies:

```bash
sudo apt-get update
sudo apt-get install build-essential cmake-gui libsuitesparse-dev \
    libboost-all-dev libeigen3-dev libgoogle-glog-dev

Clone the repository and build:

```bash
cd ~ && git clone https://github.com/adthoms/dcora.git
cd ~/dcora/
mkdir build && cd build
cmake ..
make -j$(nproc)
```

The compiled executables will be placed in `build/bin`.

## Usage

### Multi-Robot Demo (DC2-PGO)

Run distributed certifiably correct pose-graph optimization on a benchmark `.g2o` dataset:

```bash
cd build
./bin/multi-robot-example 5 ../data/smallGrid3D.g2o
```

Here, 5 specifies the number of robots.

### Single-Robot Demo (CORA)

Run certifiably correct range-aided SLAM on a benchmark `.pyfg` dataset:

```bash
cd build
./bin/single-robot-example-ra-slam ../data/tiers.pyfg
```

## Testing

To run the unit tests:

```bash
cd build
./bin/testDCORA
```

All tests must pass before submitting contributions.

## Contributing

Any contributions should pass all checks in our `.pre-commit-config.yaml` file. To install the pre-commit hooks, run `pre-commit install` in the root directory of this repository. You may need to install some dependencies to get the pre-commit hooks to work.
```bash
pip install pre-commit
sudo apt-get install cppcheck
cd ~/dcora
pre-commit install
```

## References

If you use this code in your research, please cite the following paper:

```bibtex
@article{thoms2025distributed,
  title={Distributed Certifiably Correct Range-Aided SLAM},
  author={Thoms, Alexander and Papalia, Alan and Velasquez, Jared and Rosen, David M and Narasimhan, Sriram},
  journal={arXiv preprint arXiv:2503.03192},
  year={2025}
}
```

## License

This project is licensed under the MIT License
