#!/bin/bash

# Script to download all plots from the SIMBA Alternative behavioral analysis
# This script runs the Python script to generate and save all plots

echo "🚀 Starting plot download process..."
echo "=================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed or not in PATH"
    exit 1
fi

# Check if the data file exists
if [ ! -f "data/labels.v003.000_2025-07-15 Boxes B3&B4_mjpg.analysis.csv" ]; then
    echo "❌ Data file not found. Please ensure the CSV file is in the data/ directory"
    exit 1
fi

# Run the Python script
echo "📊 Running plot generation script..."
python3 download_plots.py

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Plot download completed successfully!"
    echo "📁 All plots saved to: saved_plots/"
    echo ""
    echo "Generated plots:"
    ls -la saved_plots/*.png 2>/dev/null || echo "No plots found in saved_plots/"
else
    echo "❌ Plot generation failed. Please check the error messages above."
    exit 1
fi

echo ""
echo "🎉 All done! Your plots are ready for use."
