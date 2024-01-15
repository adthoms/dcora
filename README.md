# Distributed and Certifiably Correct Range-Aided SLAM (DCORA)


## Building

Install dependencies
```bash
sudo apt-get install build-essential cmake-gui libsuitesparse-dev libboost-all-dev libeigen3-dev libgoogle-glog-dev
```

Inside the C++ directory, execute the following commands
```bash
mkdir build
cd build
cmake ../
make
```

## Usage

The built executables are located in directory build/bin. For a minimal demo of distributed RA-SLAM on one of the benchmark datasets, inside the build directory run:
```bash
./bin/multi-robot-example 5 ../data/smallGrid3D.g2o
```

## Contributing

Any contributions should pass all checks in our `.pre-commit-config.yaml` file. To install the pre-commit hooks, run `pre-commit install` in the root directory of this repository.

You may need to install some dependencies to get the pre-commit hooks to work.

```bash
pip install pre-commit
sudo apt-get install cppcheck
cd /path/to/dcora
pre-commit install
```

## Testing

Run the unit tests via
```bash
./bin/testDCORA
```
