# OpenSpiel Windows Installation Guide

## Quick Installation (Recommended)

Simply install using pip:
```bash
pip install open-spiel
```

### Optional dependencies
For additional features like visualization and machine learning:
```bash
pip install open-spiel[full]
```

## Building from Source

If you need to build from source or contribute to the project:

### Prerequisites
- **Python 3.9+** (get from [python.org](https://python.org))
- **Git** (get from [git-scm.com](https://git-scm.com))
- **CMake 3.15+** (get from [cmake.org](https://cmake.org))
- **Visual Studio 2019 or later** with C++ development tools

### Build Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/deepmind/open_spiel.git
   cd open_spiel
   ```

2. Build the wheel:
   ```bash
   python -m pip wheel . --no-deps -w dist
   ```

3. Install the built wheel:
   ```bash
   pip install dist/open_spiel-*.whl
   ```

## Verification

Test your installation:
```python
import pyspiel

# Create a simple game
game = pyspiel.load_game("tic_tac_toe")
state = game.new_initial_state()
print("OpenSpiel is working!")
```

## Troubleshooting

### Common Issues

**"CMake not found"**
- Install CMake from [cmake.org](https://cmake.org) and add it to your PATH

**"Git not found"**  
- Install Git from [git-scm.com](https://git-scm.com) and add it to your PATH

**"MSVC compiler not found"**
- Install Visual Studio Community with C++ development tools
- Or install "Microsoft C++ Build Tools"

**"Import pyspiel failed"**
- Make sure you installed the package: `pip install open-spiel`
- Try reinstalling: `pip uninstall open-spiel && pip install open-spiel`

### Getting Help

-  [Documentation](https://openspiel.readthedocs.io/)
-  [Report Issues](https://github.com/deepmind/open_spiel/issues)
-  [Discussions](https://github.com/deepmind/open_spiel/discussions)

## Advanced Usage

### Development Installation
```bash
git clone https://github.com/deepmind/open_spiel.git
cd open_spiel
pip install -e .
```

### Custom Build Options
```bash
# Set custom CMake flags
set CMAKE_ARGS=-DCMAKE_BUILD_TYPE=Debug
python -m pip install .
```

### Using with Conda
```bash
conda create -n openspiel python=3.9
conda activate openspiel
pip install open-spiel
```

---

 **That's it!** You should now have OpenSpiel working on Windows with a simple `pip install` command.
