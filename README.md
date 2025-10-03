# Light Fields - Encoding and Quality Metrics

A comprehensive Python script for encoding light field image sequences into videos using H.264 and H.265 codecs, and evaluating their quality using PSNR, SSIM, and VMAF metrics.

## Overview

This project processes light field datasets (captured as image sequences) and:
- Encodes them into videos at multiple bitrates using H.264 and H.265 codecs
- Evaluates video quality using industry-standard metrics (PSNR, SSIM, VMAF)
- Generates comprehensive visualization graphs with quality thresholds
- Compares compression efficiency between codecs

## Features

- **Parallel Processing**: Encodes multiple videos simultaneously for faster results
- **Multiple Codecs**: Supports H.264 (libx264) and H.265 (libx265)
- **Quality Metrics**: PSNR, SSIM, and VMAF evaluation
- **Visual Analysis**: Automatic generation of quality vs bitrate graphs
- **Compression Analysis**: Efficiency comparison between codecs
- **Organized Output**: All results structured in an `output/` folder

## Requirements

- Python 3.7+
- FFmpeg (with libx264, libx265, and libvmaf support)
- Python packages:
  - pandas
  - matplotlib
  - pathlib

## Installation

1. Install FFmpeg with required libraries:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg
```

2. Install Python dependencies:
```bash
pip install pandas matplotlib
```

## Usage

1. Place your light field image sequences in folders:
   - `flowers/` - First dataset
   - `cards/` - Second dataset

2. Run the script:
```bash
python main.py
```

The script will:
- Load images from both datasets
- Create lossless reference videos
- Encode videos at multiple bitrates (500, 1000, 2000, 4000, 8000 kbps)
- Calculate quality metrics
- Generate graphs and comparison reports

## Configuration

Edit the configuration section in `main.py`:

```python
DATASETS = ["flowers", "cards"]  # Dataset folder names
BITRATES = [500, 1000, 2000, 4000, 8000]  # kbps
CODECS = {'H264': 'libx264', 'H265': 'libx265'}
PRESET = 'medium'  # Encoding speed preset
FRAMERATE = 30  # Output video framerate
```

## Output Structure

```
output/
├── encoded_videos/
│   ├── cards/
│   │   ├── H264_500kbps.mp4
│   │   ├── H264_1000kbps.mp4
│   │   ├── H265_500kbps.mp4
│   │   └── ...
│   └── flowers/
│       └── ...
├── results/
│   ├── cards_quality_metrics.csv
│   ├── flowers_quality_metrics.csv
│   └── comparison_summary.csv
└── graphs/
    ├── cards/
    │   ├── PSNR_vs_bitrate.png
    │   ├── SSIM_vs_bitrate.png
    │   ├── VMAF_vs_bitrate.png
    │   └── compression_efficiency.png
    └── flowers/
        └── ...
```

## Quality Thresholds

The script uses industry-standard quality thresholds:

### PSNR (Peak Signal-to-Noise Ratio)
- Excellent: > 38 dB
- Good: > 35 dB
- Fair: > 33 dB
- Poor: > 30 dB

### SSIM (Structural Similarity Index)
- Minimal degradation: > 0.97
- Low degradation: > 0.95

### VMAF (Video Multi-Method Assessment Fusion)
- Excellent: > 80
- Good: > 60
- Fair: > 40

## Datasets

This repository includes two example light field datasets:
- **Flowers**: 289 images (2.7MB each)
- **Cards**: 289 images (3.6MB each)

Each dataset represents a light field captured from different viewpoints.

## Performance Notes

- The script uses parallel processing (4 workers) for faster encoding
- VMAF calculation is significantly slower than PSNR/SSIM
- For testing, you can reduce datasets or disable parallel processing
- Temporary files are automatically cleaned up after processing

## Educational Context

This script was developed for an educational project at **Universidade da Beira Interior (UBI)** as part of a Multimedia course, focusing on light field video encoding and quality assessment.

## Author

**Francisco Reis**  
October 2025  
Universidade da Beira Interior (UBI)

## License

This project is available for educational and research purposes.

## Acknowledgments

This script was developed with assistance from Anthropic's Claude AI (Sonnet 4.5) through iterative prompting and refinement.

