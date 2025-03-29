# Contributing to Bumblebee

Thank you for your interest in contributing to Bumblebee! This guide will help you get started with the project and outline our contribution process.

## Code of Conduct

We're committed to providing a welcoming environment for all contributors. Please:

- Treat all participants with respect and kindness
- Provide constructive feedback
- Consider different perspectives and experiences
- Avoid offensive language and behavior

## Getting Started

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/bumblebee.git
   cd bumblebee
   ```
3. Add the original repository as an upstream remote:
   ```bash
   git remote add upstream https://github.com/socioy/bumblebee.git
   ```

### Environment Setup

#### Option 1: Using Conda (Recommended)

1. Install [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Create and activate the environment:
   ```bash
   conda env create -f environment.yml
   conda activate bumblebee
   ```

#### Option 2: Using Python venv and pip

1. Create a virtual environment:
   ```bash
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```
### Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Development Workflow

1. **Create a branch**: Create a new branch from `main` for your work
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**: Write code, add tests, and update documentation as needed

3. **Follow coding standards**: Ensure your code follows our style guide by running:
   ```bash
   black # sort python code
   isort . # sort imports
   ```

4. **Test Your Changes**

Before committing your changes, ensure your modifications work correctly. This step is critical for maintaining build integrity and overall functionality.

You can validate your work with the following steps:

1. Build the package:
   ```bash
   python setup.py sdist bdist_wheel
   ```

2. (Optional) Create a fresh virtual environment:
   - On macOS/Linux:
     ```bash
     python3 -m venv test-env
     source test-env/bin/activate
     ```
   - On Windows:
     ```bash
     python -m venv test-env
     .\test-env\Scripts\activate
     ```

3. Install the newly built package:
   ```bash
   pip install dist/bumblebee-1.0.0-py3-none-any.whl
   ```

Alternatively, use your preferred testing method to ensure everything functions as expected.

   
After building, run your tests to confirm that everything functions as expected.


5. **Commit your changes**: Use clear, descriptive commit messages:
   ```bash
   git commit -m "add: brief description of what you did"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request**: Submit a PR against the `dev` branch of the original repository

## Pull Request Guidelines

- Fill out the PR template completely
- Link any relevant issues
- Include screenshots or examples for UI changes
- Update documentation if needed
- All changes should be manually tested before submission
- PRs require approval from at least one maintainer

## Questions or Issues?

If you encounter any problems or have questions:

1. Check existing [issues](https://github.com/socioy/bumblebee/issues) first
2. If you can't find a related issue, create a new one with details about your problem or question

## Thank You!

Your contributions help make Bumblebee better for everyone. We appreciate your time and effort!