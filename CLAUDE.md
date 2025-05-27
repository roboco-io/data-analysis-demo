# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Run Commands
- Extract data: `unzip -o data/stock-market-dataset.zip -d extracted_data/`
- Run analysis: 
  - Use Python 3.7+ with pandas, numpy, matplotlib, seaborn, plotly, scipy

## Code Style Guidelines
- Follow PEP 8 conventions
- Use snake_case for function and variable names
- Include docstrings with Parameters and Returns sections
- 4-space indentation
- Use meaningful variable names
- Write descriptive comments for complex operations
- Prefer f-strings for string formatting
- Organize imports: standard library, third-party, local modules

## Error Handling
- Use try/except blocks for file operations
- Handle missing data with appropriate pandas methods
- Validate inputs in functions with descriptive error messages