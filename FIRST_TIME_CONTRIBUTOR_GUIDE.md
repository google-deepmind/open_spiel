# 🎮 First-Time Contributor Guide to OpenSpiel

Welcome! This guide will help you understand the OpenSpiel codebase and how to make your first contribution. OpenSpiel is a framework for reinforcement learning research in games, and we're excited to have you join our community!

## 📚 Table of Contents

1. [What is OpenSpiel?](#what-is-openspiel)
2. [Understanding the Architecture](#understanding-the-architecture)
3. [Codebase Structure](#codebase-structure)
4. [Core Concepts](#core-concepts)
5. [Setting Up Your Development Environment](#setting-up-your-development-environment)
6. [Making Your First Contribution](#making-your-first-contribution)
7. [Common Contribution Types](#common-contribution-types)
8. [Code Style and Best Practices](#code-style-and-best-practices)
9. [Testing Your Changes](#testing-your-changes)
10. [Submitting Your Pull Request](#submitting-your-pull-request)

---

## 🎯 What is OpenSpiel?

OpenSpiel is a collection of environments and algorithms for research in:
- **Reinforcement Learning** in games
- **Game theory** and strategic reasoning
- **Multi-agent systems**

### Key Features:
- 🎲 **80+ games** implemented (chess, poker, Go, tic-tac-toe, etc.)
- 🤖 **Multiple AI algorithms** (MCTS, CFR, AlphaZero, etc.)
- 🔄 Supports various game types:
  - Zero-sum, cooperative, and general-sum games
  - Perfect and imperfect information
  - Turn-based and simultaneous move games
- 🐍 **Dual API**: C++ (performance) and Python (ease of use)

---

## 🏗️ Understanding the Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         OpenSpiel Framework                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌───────────────┐         ┌──────────────┐                    │
│  │   Python API   │◄───────►│   C++ Core   │                    │
│  │  (pyspiel)     │ binding │  (libspiel)  │                    │
│  └───────┬────────┘         └──────┬───────┘                    │
│          │                          │                            │
│          │                          │                            │
│  ┌───────▼──────────────────────────▼────────────────────┐     │
│  │              Game Interface (spiel.h)                  │     │
│  │  • Game: High-level game description                   │     │
│  │  • State: Specific point in game trajectory            │     │
│  └────────────────────────────────────────────────────────┘     │
│          │                          │                            │
│          │                          │                            │
│  ┌───────▼────────┐        ┌───────▼─────────┐                 │
│  │  Algorithms    │        │     Games       │                 │
│  │                │        │                 │                 │
│  │  • MCTS        │        │  • tic_tac_toe  │                 │
│  │  • CFR         │        │  • chess        │                 │
│  │  • AlphaZero   │        │  • poker        │                 │
│  │  • Minimax     │        │  • go           │                 │
│  │  • Q-Learning  │        │  • 80+ more...  │                 │
│  └────────────────┘        └─────────────────┘                 │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### The Two Main Concepts

```
┌─────────────────────────────────────────────────────────────────┐
│                        Game vs State                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌────────────────────────────────────────────┐                 │
│  │  GAME: The Rules and Structure              │                 │
│  │  ────────────────────────────────────       │                 │
│  │  • Number of players                        │                 │
│  │  • Game type (turn-based, simultaneous)     │                 │
│  │  • Utility bounds (min/max scores)          │                 │
│  │  • Creates initial states                   │                 │
│  │                                              │                 │
│  │  Example: "The game of Tic-Tac-Toe"        │                 │
│  └────────────────────────────────────────────┘                 │
│                          │                                        │
│                          │ creates                                │
│                          ▼                                        │
│  ┌────────────────────────────────────────────┐                 │
│  │  STATE: A Specific Game Position            │                 │
│  │  ────────────────────────────────────       │                 │
│  │  • Current board/cards/position             │                 │
│  │  • Whose turn it is                         │                 │
│  │  • Legal actions available                  │                 │
│  │  • Can apply actions to reach new states    │                 │
│  │                                              │                 │
│  │  Example: "X has marked center square"      │                 │
│  └────────────────────────────────────────────┘                 │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Game Tree Representation

Every game in OpenSpiel is represented as a tree:

```
                        Initial State
                       (Empty board)
                            │
              ┌─────────────┼─────────────┐
              │             │             │
         Action: X→0   Action: X→1   Action: X→2
              │             │             │
              ▼             ▼             ▼
         State: [X,_,_] State: [_,X,_] State: [_,_,X]
              │             │             │
         ┌────┴───┐    ┌────┴───┐    ┌────┴───┐
         │        │    │        │    │        │
    Action: O  Action: O    (and so on...)
         │        │
         ▼        ▼
    [X,O,_]   [X,_,O]
        ...      ...
```

**Key Points:**
- 🌳 **Nodes** = States (game positions)
- ➡️ **Edges** = Actions (moves)
- 🎲 **Chance nodes** = Random events (card dealing, dice rolls)

---

## 📂 Codebase Structure

### Root Directory Layout

```
open_spiel/
├── 📄 README.md                    # Project overview
├── 📄 CONTRIBUTING.md              # Contribution guidelines
├── 📄 install.sh                   # Installation script
├── 📄 setup.py                     # Python package setup
├── 📄 requirements.txt             # Python dependencies
│
├── 📁 docs/                        # Documentation files
│   ├── install.md
│   ├── concepts.md
│   ├── developer_guide.md
│   └── ...
│
└── 📁 open_spiel/                  # Main source code
    ├── 📄 spiel.h                  # Core API definition
    ├── 📄 spiel.cc                 # Core implementation
    │
    ├── 📁 games/                   # Game implementations
    │   ├── tic_tac_toe/
    │   ├── chess/
    │   ├── poker/
    │   └── ...                     # 80+ games!
    │
    ├── 📁 algorithms/              # AI algorithms
    │   ├── mcts.cc/h               # Monte Carlo Tree Search
    │   ├── cfr.cc/h                # Counterfactual Regret
    │   ├── minimax.cc/h            # Minimax algorithm
    │   └── ...
    │
    ├── 📁 python/                  # Python bindings
    │   ├── algorithms/             # Python algorithms
    │   ├── games/                  # Python games
    │   ├── examples/               # Example scripts
    │   └── tests/                  # Python tests
    │
    ├── 📁 examples/                # C++ example programs
    ├── 📁 tests/                   # C++ test utilities
    └── 📁 scripts/                 # Build & utility scripts
```

### Key Directories Explained

| Directory | Purpose | When to Use |
|-----------|---------|-------------|
| `open_spiel/games/` | C++ game implementations | Adding a new game in C++ |
| `open_spiel/algorithms/` | C++ algorithm implementations | Adding new C++ algorithms |
| `open_spiel/python/games/` | Python game implementations | Adding a Python-only game |
| `open_spiel/python/algorithms/` | Python algorithm implementations | Adding Python algorithms |
| `open_spiel/examples/` | C++ example usage | Learning how to use the API |
| `open_spiel/python/examples/` | Python example usage | Learning Python API |

---

## 🧠 Core Concepts

### 1. The Game Class

The `Game` class represents the **rules** of a game:

```cpp
// Key methods in the Game class:
class Game {
  // Get basic information
  int NumPlayers();                    // How many players?
  int MaxGameLength();                 // Maximum moves?
  double MinUtility() / MaxUtility(); // Score bounds?
  
  // Create states
  std::unique_ptr<State> NewInitialState();  // Start a new game
  
  // Game properties
  GameType GetType();                  // Game characteristics
};
```

**Think of it as:** The game box and rulebook 📦

### 2. The State Class

The `State` class represents a **specific position** in a game:

```cpp
// Key methods in the State class:
class State {
  // Query current state
  Player CurrentPlayer();              // Whose turn is it?
  bool IsTerminal();                   // Is the game over?
  std::vector<double> Returns();       // Final scores (if terminal)
  
  // Get possible moves
  std::vector<Action> LegalActions();  // What moves are available?
  
  // Modify state
  void ApplyAction(Action action);     // Make a move!
  std::unique_ptr<State> Clone();      // Copy this state
  
  // Display
  std::string ToString();              // Human-readable state
};
```

**Think of it as:** A snapshot of the game board at a specific moment 📸

### 3. Game Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Typical Game Loop                             │
└─────────────────────────────────────────────────────────────────┘

    ┌──────────────────┐
    │ Load Game        │
    │ game = LoadGame  │
    │  ("tic_tac_toe") │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ Create State     │
    │ state = game.    │
    │  NewInitialState │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ Is Terminal?     │◄──────────────┐
    └────────┬─────────┘               │
             │ NO                       │
             ▼                          │
    ┌──────────────────┐               │
    │ Get Legal Actions│               │
    │ actions = state. │               │
    │  LegalActions()  │               │
    └────────┬─────────┘               │
             │                          │
             ▼                          │
    ┌──────────────────┐               │
    │ Choose Action    │               │
    │ (AI or human)    │               │
    └────────┬─────────┘               │
             │                          │
             ▼                          │
    ┌──────────────────┐               │
    │ Apply Action     │               │
    │ state.Apply      │               │
    │  Action(action)  │───────────────┘
    └──────────────────┘
             │ YES
             ▼
    ┌──────────────────┐
    │ Get Final Scores │
    │ state.Returns()  │
    └──────────────────┘
             │
             ▼
    ┌──────────────────┐
    │      END         │
    └──────────────────┘
```

---

## 🛠️ Setting Up Your Development Environment

### Step 1: Install Prerequisites

#### On Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install cmake clang python3-dev git
```

#### On MacOS:
```bash
brew install cmake python3
# Install Xcode command-line tools
xcode-select --install
```

### Step 2: Clone the Repository

```bash
git clone https://github.com/google-deepmind/open_spiel.git
cd open_spiel

### Step 3: Run Installation Script

```bash
# This downloads dependencies and sets up the environment
./install.sh
```

### Step 4: Build the Project

```bash
# Build C++ and Python components
./open_spiel/scripts/build_and_run_tests.sh
```

### Step 5: Set Up Python Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install OpenSpiel in development mode
pip install -e .

# Set up PYTHONPATH (IMPORTANT!)
# This ensures Python can find OpenSpiel modules
export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)/build/python

# To make PYTHONPATH permanent in your virtual environment,
# add these lines to venv/bin/activate (after the 'export PATH' line):
echo 'export PYTHONPATH=$PYTHONPATH:'$(pwd) >> venv/bin/activate
echo 'export PYTHONPATH=$PYTHONPATH:'$(pwd)'/build/python' >> venv/bin/activate

# Now PYTHONPATH will be set automatically every time you activate the venv!
```

**Important Note:** If you skip setting `PYTHONPATH`, you'll encounter import errors like `ModuleNotFoundError: No module named 'pyspiel'` or `ImportError: No module named 'open_spiel'`. The PYTHONPATH must point to:
1. The repository root (for `open_spiel` package)
2. The `build/python` directory (for compiled C++ bindings like `pyspiel`)

### Step 6: Test Your Installation

```bash
# Test C++ example
./build/examples/example --game=tic_tac_toe

# Test Python
python3 open_spiel/python/examples/example.py --game_string=tic_tac_toe
```

✅ If these work, you're all set!

---

## 🚀 Making Your First Contribution

### Contribution Workflow Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                   Contribution Workflow                         │
└────────────────────────────────────────────────────────────────┘

    ┌──────────────────┐
    │ 1. Find an Issue │
    │  or Idea         │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ 2. Fork & Clone  │
    │  Repository      │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ 3. Create Branch │
    │  git checkout -b │
    │  feature-branch  │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ 4. Make Changes  │
    │  (Code, test,    │
    │   document)      │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ 5. Run Tests     │
    │  & Linters       │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ 6. Commit &      │
    │  Push Changes    │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ 7. Create Pull   │
    │  Request (PR)    │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ 8. Code Review   │
    │  & Feedback      │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ 9. Merge! 🎉     │
    └──────────────────┘
```

### Step-by-Step Guide

#### 1. Find Something to Work On

Browse the [Issues page](https://github.com/deepmind/open_spiel/issues) and look for:
- 🏷️ **"help wanted"** - Maintenance tasks
- 🏷️ **"contribution welcome"** - New features
- 🏷️ **"good first issue"** - Beginner-friendly tasks

Or propose your own idea by opening an issue first!

#### 2. Fork the Repository

Click the "Fork" button on GitHub to create your own copy.

#### 3. Create a Feature Branch

```bash
git checkout -b my-new-feature
```

Use descriptive branch names like:
- `add-connect-four-game`
- `fix-poker-reset-bug`
- `improve-mcts-documentation`

#### 4. Make Your Changes

See the next section for specific contribution types.

#### 5. Test Your Changes

```bash
# Build and run all tests
./open_spiel/scripts/build_and_run_tests.sh

# Or run specific tests
cd build
ctest --verbose
```

#### 6. Commit Your Changes

```bash
git add .
git commit -m "Add Connect Four game implementation"
```

**Good commit messages:**
- ✅ "Add Connect Four game with tests and playthrough"
- ✅ "Fix reset bug in poker game state"
- ❌ "Fixed stuff" (too vague)
- ❌ "asdfg" (not descriptive)

#### 7. Push and Create Pull Request

```bash
git push origin my-new-feature
```

Then go to GitHub and click "Create Pull Request".

**In your PR description, include:**
- What changes you made
- Why you made them
- How to test them
- Screenshots/examples if applicable

---

## 🎨 Common Contribution Types

### Type 1: Adding a New Game 🎲

This is one of the most common contributions! Here's how:

#### Game Implementation Structure

```
open_spiel/games/my_game/
├── CMakeLists.txt          # Build configuration
├── my_game.h              # Header file
├── my_game.cc             # Implementation
└── my_game_test.cc        # Unit tests
```

#### Step-by-Step Process

**1. Choose a Template Game**

Pick a similar game to copy from:
- Simple perfect info → `tic_tac_toe`
- With chance events → `backgammon` or `pig`
- Simultaneous moves → `goofspiel`
- Imperfect info → `leduc_poker`

**2. Copy the Template**

```bash
cd open_spiel/games
cp -r tic_tac_toe my_game
```

**3. Update File Names and Content**

```bash
cd my_game
mv tic_tac_toe.h my_game.h
mv tic_tac_toe.cc my_game.cc
mv tic_tac_toe_test.cc my_game_test.cc
```

**4. Modify the Code**

In `my_game.h`:
```cpp
#ifndef OPEN_SPIEL_GAMES_MY_GAME_H_
#define OPEN_SPIEL_GAMES_MY_GAME_H_

namespace open_spiel {
namespace my_game {

// Game constants
inline constexpr int kNumPlayers = 2;
// ... other constants

class MyGameState : public State {
 public:
  // Implement required methods
  Player CurrentPlayer() const override;
  std::vector<Action> LegalActions() const override;
  void DoApplyAction(Action action) override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string ToString() const override;
  
  // ... your custom state data
};

class MyGameGame : public Game {
 public:
  // Implement game properties
  int NumPlayers() const override { return kNumPlayers; }
  int MaxGameLength() const override { /* ... */ }
  std::unique_ptr<State> NewInitialState() const override;
  // ...
};

}  // namespace my_game
}  // namespace open_spiel

#endif
```

**5. Register Your Game**

In `my_game.cc`:
```cpp
namespace {
constexpr char kGameShortName[] = "my_game";
constexpr char kGameLongName[] = "My Amazing Game";
}

REGISTER_SPIEL_GAME(kGameShortName, MyGameGame);
```

**6. Add to CMake**

Edit `open_spiel/games/CMakeLists.txt`:
```cmake
add_library(my_game OBJECT
  my_game/my_game.cc
  my_game/my_game.h
)
target_link_libraries(my_game PUBLIC game_parameters)

add_executable(my_game_test my_game/my_game_test.cc)
target_link_libraries(my_game_test my_game)
add_test(my_game_test my_game_test)
```

**7. Write Tests**

In `my_game_test.cc`:
```cpp
#include "open_spiel/games/my_game/my_game.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace my_game {
namespace {

void BasicTests() {
  testing::LoadGameTest("my_game");
  testing::RandomSimTest(*LoadGame("my_game"), 100);
}

}  // namespace
}  // namespace my_game
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::my_game::BasicTests();
  return 0;
}
```

**8. Generate Playthrough**

```bash
./open_spiel/scripts/generate_new_playthrough.sh my_game
```

This creates a regression test file.

**9. Test Everything**

```bash
./open_spiel/scripts/build_and_run_tests.sh
```

### Type 2: Adding a New Algorithm 🤖

Algorithms go in `open_spiel/algorithms/` (C++) or `open_spiel/python/algorithms/` (Python).

#### Structure

```cpp
// my_algorithm.h
#ifndef OPEN_SPIEL_ALGORITHMS_MY_ALGORITHM_H_
#define OPEN_SPIEL_ALGORITHMS_MY_ALGORITHM_H_

namespace open_spiel {
namespace algorithms {

class MyAlgorithm {
 public:
  MyAlgorithm(const Game& game);
  
  // Main algorithm logic
  Action ComputeAction(const State& state);
  
 private:
  // Internal state
};

}  // namespace algorithms
}  // namespace open_spiel

#endif
```

### Type 3: Fixing Bugs 🐛

1. **Find the bug** - Look in Issues or find it yourself
2. **Reproduce it** - Create a minimal test case
3. **Fix it** - Make the smallest change possible
4. **Test it** - Add a test to prevent regression
5. **Document it** - Explain what was wrong and how you fixed it

### Type 4: Improving Documentation 📚

Documentation improvements are always welcome!

- Fix typos or unclear explanations
- Add examples
- Improve code comments
- Update README files

---

## ✨ Code Style and Best Practices

### C++ Style

Follow [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)

**Key Points:**
```cpp
// Use snake_case for functions and variables
void my_function() {
  int my_variable = 42;
}

// Use PascalCase for classes
class MyClass {
 public:
  MyClass();  // Constructor
  
 private:
  int member_variable_;  // Member variables end with _
};

// Use k prefix for constants
constexpr int kMaxPlayers = 4;

// Use descriptive names
✅ int num_players = 2;
❌ int n = 2;
```

**Run linter before committing:**
```bash
cpplint open_spiel/games/my_game/*.cc
```

### Python Style

Follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

**Key Points:**
```python
# Use snake_case for everything except classes
def my_function():
    my_variable = 42

# Use PascalCase for classes
class MyClass:
    def __init__(self):
        self.member_variable = 0

# Use UPPER_CASE for constants
MAX_PLAYERS = 4

# Add docstrings
def my_function(param):
    """Does something useful.
    
    Args:
        param: Description of parameter.
        
    Returns:
        Description of return value.
    """
    return param * 2
```

**Run linter before committing:**
```bash
pylint open_spiel/python/games/my_game.py
```

### General Best Practices

1. **Keep it simple** - Prefer clarity over cleverness
2. **Write tests** - Every feature needs tests
3. **Add comments** - Explain *why*, not just *what*
4. **Avoid dependencies** - Don't add new libraries unless necessary
5. **Document public APIs** - Help others understand your code

---

## 🧪 Testing Your Changes

### Test Levels

```
┌────────────────────────────────────────────────────────────┐
│                      Testing Pyramid                        │
└────────────────────────────────────────────────────────────┘

                     ▲
                    ╱ ╲
                   ╱   ╲
                  ╱ E2E ╲        ← Full integration tests
                 ╱───────╲
                ╱         ╲
               ╱Integration╲     ← Cross-component tests
              ╱─────────────╲
             ╱               ╲
            ╱   Unit Tests    ╲  ← Individual function tests
           ╱───────────────────╲
          ╱                     ╲
         ╱  Component Tests      ╲ ← Game/Algorithm tests
        ╱─────────────────────────╲
       ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
```

### Running Tests

#### All Tests
```bash
./open_spiel/scripts/build_and_run_tests.sh
```

#### Specific Test
```bash
cd build
./games/my_game_test
```

#### Python Tests
```bash
python3 -m pytest open_spiel/python/tests/
```

### Writing Good Tests

**For Games:**
```cpp
void BasicGameTests() {
  // Load test
  testing::LoadGameTest("my_game");
  
  // Random simulation test (plays 100 random games)
  testing::RandomSimTest(*LoadGame("my_game"), 100);
  
  // Serialization test
  testing::SerializeDeserializeTest(*LoadGame("my_game"));
}
```

**For Algorithms:**
```cpp
TEST(MyAlgorithmTest, BasicFunctionality) {
  auto game = LoadGame("tic_tac_toe");
  MyAlgorithm algo(*game);
  
  auto state = game->NewInitialState();
  Action action = algo.ComputeAction(*state);
  
  // Verify action is legal
  EXPECT_TRUE(std::find(state->LegalActions().begin(),
                        state->LegalActions().end(),
                        action) != state->LegalActions().end());
}
```

---

## 📤 Submitting Your Pull Request

### Before Submitting Checklist

- [ ] Code compiles without warnings
- [ ] All tests pass
- [ ] Code follows style guide (run linters)
- [ ] Added tests for new functionality
- [ ] Updated documentation if needed
- [ ] Commit messages are descriptive
- [ ] Generated playthrough for new games
- [ ] No unnecessary dependencies added

### Pull Request Template

```markdown
## Description
Brief description of what this PR does.

## Type of Change
- [ ] Bug fix
- [ ] New feature (game, algorithm, etc.)
- [ ] Documentation update
- [ ] Code refactoring

## Testing
- [ ] All existing tests pass
- [ ] Added new tests for new functionality
- [ ] Manually tested the changes

## How to Test
1. Build the project
2. Run `./build/examples/example --game=my_game`
3. Verify output matches expected behavior

## Related Issues
Closes #123 (if applicable)

## Additional Notes
Any other context or screenshots.
```

### After Submitting

1. **Respond to feedback** - Reviewers may request changes
2. **Be patient** - Reviews are typically done within 1-2 weeks
3. **Make requested changes** - Push new commits to your branch
4. **Celebrate when merged!** 🎉

---

## 🎓 Learning Resources

### Official Documentation
- [API Reference](docs/api_reference.md)
- [Developer Guide](docs/developer_guide.md)
- [Games Overview](docs/games.md)
- [Algorithms Overview](docs/algorithms.md)

### Tutorials
- [Video Tutorial by Marc Lanctot](https://www.youtube.com/watch?v=8NCPqtPwlFQ)
- [CFR and REINFORCE Tutorial](https://www.youtube.com/watch?v=o6JNHoGUXCo)
- [Google Colab Notebooks](open_spiel/colabs/)

### Code Examples
- C++ Examples: `open_spiel/examples/`
- Python Examples: `open_spiel/python/examples/`

### Reference Games to Study
- **Simple:** `tic_tac_toe` - Clean, easy to understand
- **Medium:** `breakthrough` - More complex state
- **Complex:** `chess` - Full-featured implementation
- **Imperfect Info:** `leduc_poker` - Hidden information
- **Simultaneous:** `goofspiel` - Both players move at once

---

## 🤝 Community Guidelines

### Be Respectful
- Follow [Google's Open Source Community Guidelines](https://opensource.google.com/conduct/)
- Be kind and constructive in reviews
- Help others learn and grow

### Communication Channels
- **GitHub Issues** - Bug reports and feature requests
- **Pull Requests** - Code contributions
- **Discussions** - General questions and ideas

### Getting Help
- Check existing documentation first
- Search closed issues for similar problems
- Open a new issue with detailed description
- Include code examples and error messages

---

## 📋 Quick Reference

### Common Commands

```bash
# Setup
./install.sh
./open_spiel/scripts/build_and_run_tests.sh

# Development
git checkout -b my-feature
# ... make changes ...
git add .
git commit -m "Description"
git push origin my-feature

# Testing
cd build
ctest --verbose
./games/my_game_test

# Linting
cpplint open_spiel/games/my_game/*.cc
pylint open_spiel/python/games/my_game.py

# Play a game
./build/examples/example --game=my_game
python3 open_spiel/python/examples/example.py --game_string=my_game
```

### Important Files

| File | Purpose |
|------|---------|
| `open_spiel/spiel.h` | Core API definition |
| `open_spiel/games/CMakeLists.txt` | Build config for games |
| `open_spiel/python/games/__init__.py` | Python game registry |
| `docs/developer_guide.md` | Detailed dev docs |

---

## 🎉 Final Words

Contributing to OpenSpiel is a great way to:
- Learn about game theory and AI
- Improve your C++/Python skills
- Join a vibrant research community
- Make an impact on AI research

**Remember:**
- Start small - even documentation improvements help!
- Ask questions - the community is friendly
- Be patient - quality takes time
- Have fun - you're building cool AI stuff! 🤖

---

## 📬 Getting Started Today

1. ⭐ Star the repository
2. 🍴 Fork it to your account
3. 💻 Clone and set up locally
4. 🔍 Find a good first issue
5. 🚀 Make your first contribution!

**Welcome to the OpenSpiel community! We can't wait to see what you build.** 🎮✨

---

*Last Updated: October 2025*  
*For questions or issues with this guide, please open an issue on GitHub.*
