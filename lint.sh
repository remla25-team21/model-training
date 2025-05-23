#!/bin/bash
set -euo pipefail

# Define the target directory
TARGET_DIR="src"

# Run Pylint and redirect output to pylint_output.txt
echo "Running Pylint..."
pylint "$TARGET_DIR" > pylint_output.txt || true
echo "Pylint report saved to pylint_output.txt"

# Run Flake8 and redirect output to flake8_output.txt
echo "Running Flake8..."
flake8 "$TARGET_DIR" > flake8_output.txt || true
echo "Flake8 report saved to flake8_output.txt"

# Run Bandit and redirect output to bandit_output.txt
echo "Running Bandit..."
bandit -r "$TARGET_DIR" --ini .bandit > bandit_output.txt || true
echo "Bandit report saved to bandit_output.txt"

echo ""
echo "=== Linting Summary ==="

# Extract Pylint quality score
PYLINT_SCORE_LINE=$(grep "Your code has been rated at" pylint_output.txt || true)
if [[ -n "$PYLINT_SCORE_LINE" ]]; then
    PYLINT_SCORE=$(echo "$PYLINT_SCORE_LINE" | awk '{print $7}')
    echo "Pylint Quality Score: $PYLINT_SCORE"
else
    echo "Pylint Quality Score: Unable to determine (check pylint_output.txt)"
fi

# Count Flake8 issues
FLAKE8_ISSUES=$(wc -l < flake8_output.txt)
echo "Flake8 Issues: $FLAKE8_ISSUES"

echo ""
echo "Linting completed."
