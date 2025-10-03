"""
Light Fields - Encoding and Quality Metrics

Light Field Video Encoding and Quality Evaluation
Processes both flowers and cards datasets with parallel encoding (for faster results)
Includes quality threshold lines and comprehensive comparison

This script processes are heavy for the system
You can modify the code to proccess one thing at a time, instead of using parallel processing
Use only one and smaller dataset to reduce the load on the system

This script was generated via multi-prompt chatting with the flagship Anthropic model, Claude Sonnet 4.5 Thinking
It was used in an educational context with the purpose of writing a report about the topic

Francisco Reis, Oct 2025 Universidade da Beira Interior (UBI)
"""

import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASETS = ["flowers", "cards"]
BASE_OUTPUT_FOLDER = "output/encoded_videos"
BASE_RESULTS_FOLDER = "output/results"
BASE_GRAPHS_FOLDER = "output/graphs"
TEMP_FOLDER_BASE = "output/temp_frames"

# Encoding parameters
BITRATES = [500, 1000, 2000, 4000, 8000]  # kbps
CODECS = {'H264': 'libx264', 'H265': 'libx265'}
PRESET = 'medium'
FRAMERATE = 30

# Quality thresholds for visualization
PSNR_THRESHOLDS = {'Excellent': 38, 'Good': 35, 'Fair': 33, 'Poor': 30}
SSIM_THRESHOLDS = {'Minimal': 0.97, 'Low': 0.95}
VMAF_THRESHOLDS = {'Excellent': 80, 'Good': 60, 'Fair': 40}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_directories(dataset_name):
    """Create organized output directories for a dataset."""
    folders = [
        os.path.join(BASE_OUTPUT_FOLDER, dataset_name),
        BASE_RESULTS_FOLDER,
        os.path.join(BASE_GRAPHS_FOLDER, dataset_name),
        f"{TEMP_FOLDER_BASE}_{dataset_name}"
    ]
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
    return folders

def get_image_list(dataset_path):
    """Get sorted list of image files from dataset folder."""
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG']
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(dataset_path, ext)))
    images.sort()
    return images

def prepare_image_sequence(image_list, temp_folder):
    """Create sequential symlinks for FFmpeg."""
    # Clean temp folder
    for f in glob.glob(os.path.join(temp_folder, "*")):
        try:
            os.remove(f)
        except:
            pass
    
    first_ext = os.path.splitext(image_list[0])[1]
    
    # Create sequential symlinks
    for i, img_path in enumerate(image_list, start=1):
        link_name = os.path.join(temp_folder, f"frame_{i:04d}{first_ext}")
        rel_path = os.path.relpath(img_path, temp_folder)
        try:
            os.symlink(rel_path, link_name)
        except OSError:
            import shutil
            shutil.copy(img_path, link_name)
    
    return os.path.join(temp_folder, f"frame_%04d{first_ext}")

def get_file_size_mb(filepath):
    """Get file size in megabytes."""
    return os.path.getsize(filepath) / (1024 * 1024)

# ============================================================================
# ENCODING FUNCTIONS
# ============================================================================

def encode_video(input_pattern, output_path, codec, bitrate, framerate=30, preset='medium'):
    """Encode video using FFmpeg."""
    cmd = [
        'ffmpeg', '-y',                    # Overwrite output files without asking
        '-framerate', str(framerate),      # Set input frame rate (fps) for the image sequence
        '-i', input_pattern,               # Input file pattern (e.g., "frames_%03d.jpg")
        '-c:v', codec,                    # Video codec (e.g., "libx264" for H.264, "libx265" for H.265)
        '-b:v', f'{bitrate}k',            # Video bitrate in kbps (e.g., "1000k" for 1000 kbps)
        '-preset', preset,                # Encoding speed vs compression efficiency preset (slow, medium, fast, etc.)
        '-pix_fmt', 'yuv420p',            # Pixel format - ensures compatibility with most players
        '-movflags', '+faststart',        # Optimize for web streaming (moves metadata to beginning)
        output_path                        # Output video file path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
        return True
    except:
        return False

def create_reference_video(input_pattern, output_folder):
    """Create lossless reference video."""
    reference_path = os.path.join(output_folder, 'reference_lossless.mp4')
    
    cmd = [
        'ffmpeg', '-y',                    # Overwrite output files without asking
        '-framerate', str(FRAMERATE),      # Set input frame rate (fps) for the image sequence
        '-i', input_pattern,               # Input file pattern (e.g., "frames_%03d.jpg")
        '-c:v', 'libx264',                # Use H.264 codec for lossless encoding
        '-qp', '0',                       # Quantization parameter 0 = lossless compression
        '-preset', 'veryslow',            # Slowest preset for maximum compression efficiency
        '-pix_fmt', 'yuv420p',            # Pixel format - ensures compatibility with most players
        reference_path                     # Output reference video file path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1200)
        return reference_path
    except:
        return None

def encode_single_video_task(args):
    """Task function for parallel encoding."""
    input_pattern, output_path, codec_lib, bitrate = args
    success = encode_video(input_pattern, output_path, codec_lib, bitrate, FRAMERATE, PRESET)
    if success:
        return {'path': output_path, 'size_mb': get_file_size_mb(output_path), 'success': True}
    return {'success': False}

def encode_all_videos_parallel(input_pattern, output_folder, dataset_name):
    """Encode all videos in parallel using multiple workers."""
    tasks = []
    video_info = []
    
    # Prepare all tasks
    for codec_name, codec_lib in CODECS.items():
        for bitrate in BITRATES:
            output_filename = f'{codec_name}_{bitrate}kbps.mp4'
            output_path = os.path.join(output_folder, output_filename)
            tasks.append((input_pattern, output_path, codec_lib, bitrate))
            video_info.append({'codec': codec_name, 'bitrate': bitrate, 'path': output_path})
    
    print(f"  Starting parallel encoding of {len(tasks)} videos...")
    
    encoded_videos = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(encode_single_video_task, task): idx 
                  for idx, task in enumerate(tasks)}
        
        completed = 0
        for future in as_completed(futures):
            idx = futures[future]
            result = future.result()
            completed += 1
            
            if result['success']:
                info = video_info[idx]
                info['size_mb'] = result['size_mb']
                encoded_videos.append(info)
                print(f"    [{completed}/{len(tasks)}] ✓ {os.path.basename(info['path'])} ({info['size_mb']:.1f} MB)")
            else:
                print(f"    [{completed}/{len(tasks)}] ✗ Encoding failed")
    
    return encoded_videos

# ============================================================================
# QUALITY METRIC FUNCTIONS
# ============================================================================

def calculate_psnr(reference_video, distorted_video):
    """Calculate PSNR between reference and distorted video."""
    cmd = [
        'ffmpeg', '-i', distorted_video,    # First input: the compressed/distorted video
        '-i', reference_video,              # Second input: the reference/lossless video
        '-filter_complex', 'psnr',          # Apply PSNR filter to compare the two videos
        '-f', 'null',                       # Output format: null (no actual output file)
        '-'                                 # Output to stdout (discarded, we only need stderr)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        stderr = result.stderr
        
        match = re.search(r'PSNR.*?average:([0-9.]+)', stderr)
        if match:
            return float(match.group(1))
        
        match = re.search(r'psnr_avg:([0-9.]+)', stderr)
        if match:
            return float(match.group(1))
        
        return None
    except:
        return None

def calculate_ssim(reference_video, distorted_video):
    """Calculate SSIM between reference and distorted video."""
    cmd = [
        'ffmpeg', '-i', distorted_video,    # First input: the compressed/distorted video
        '-i', reference_video,              # Second input: the reference/lossless video
        '-filter_complex', 'ssim',          # Apply SSIM filter to compare the two videos
        '-f', 'null',                       # Output format: null (no actual output file)
        '-'                                 # Output to stdout (discarded, we only need stderr)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        stderr = result.stderr
        
        match = re.search(r'All:([0-9.]+)', stderr)
        if match:
            return float(match.group(1))
        
        return None
    except:
        return None

def calculate_vmaf(reference_video, distorted_video):
    """Calculate VMAF between reference and distorted video."""
    # Check if libvmaf is available
    try:
        check = subprocess.run(['ffmpeg', '-filters'], capture_output=True, text=True)
        if 'libvmaf' not in check.stdout:
            return None
    except:
        return None
    
    cmd = [
        'ffmpeg', '-i', distorted_video,    # First input: the compressed/distorted video
        '-i', reference_video,              # Second input: the reference/lossless video
        '-filter_complex', '[0:v][1:v]libvmaf',  # Apply VMAF filter comparing video streams
        '-f', 'null',                       # Output format: null (no actual output file)
        '-'                                 # Output to stdout (discarded, we only need stderr)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        stderr = result.stderr
        
        match = re.search(r'VMAF score[:\s]+([0-9.]+)', stderr, re.IGNORECASE)
        if match:
            return float(match.group(1))
        
        match = re.search(r'vmaf.*?mean:([0-9.]+)', stderr, re.IGNORECASE)
        if match:
            return float(match.group(1))
        
        return None
    except:
        return None

def evaluate_videos(encoded_videos, reference_video, dataset_name):
    """Evaluate all encoded videos with quality metrics."""
    results = []
    total = len(encoded_videos)
    
    print(f"  Starting quality evaluation for {total} videos...")
    print(f"  (Note: VMAF takes longer than PSNR/SSIM)")
    
    for idx, video_info in enumerate(encoded_videos, 1):
        codec = video_info['codec']
        bitrate = video_info['bitrate']
        
        print(f"    [{idx}/{total}] {codec} @ {bitrate} kbps...", end=" ", flush=True)
        
        psnr = calculate_psnr(reference_video, video_info['path'])
        ssim = calculate_ssim(reference_video, video_info['path'])
        vmaf = calculate_vmaf(reference_video, video_info['path'])
        
        print(f"PSNR: {psnr:.2f}" if psnr else "PSNR: N/A", end=" | ")
        print(f"SSIM: {ssim:.4f}" if ssim else "SSIM: N/A", end=" | ")
        print(f"VMAF: {vmaf:.2f}" if vmaf else "VMAF: N/A")
        
        results.append({
            'Dataset': dataset_name,
            'Codec': codec,
            'Bitrate (kbps)': bitrate,
            'File Size (MB)': round(video_info['size_mb'], 2),
            'PSNR (dB)': round(psnr, 2) if psnr else None,
            'SSIM': round(ssim, 4) if ssim else None,
            'VMAF': round(vmaf, 2) if vmaf else None
        })
    
    return pd.DataFrame(results)

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def generate_graphs_with_thresholds(df, dataset_name, output_folder):
    """Generate quality graphs with threshold lines."""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # ========== PSNR GRAPH ==========
    if 'PSNR (dB)' in df.columns and not df['PSNR (dB)'].isna().all():
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for codec in df['Codec'].unique():
            data = df[df['Codec'] == codec].dropna(subset=['PSNR (dB)'])
            data = data.sort_values('Bitrate (kbps)') 
            if not data.empty:
                ax.plot(data['Bitrate (kbps)'], data['PSNR (dB)'], 
                       marker='o', label=f'{codec}', linewidth=2.5, markersize=10)
        
        threshold_colors = {'Excellent': 'green', 'Good': 'yellowgreen', 'Fair': 'orange', 'Poor': 'red'}
        for label, threshold in PSNR_THRESHOLDS.items():
            ax.axhline(y=threshold, color=threshold_colors[label], linestyle='--', 
                      linewidth=1.5, alpha=0.7, label=f'{label} ({threshold} dB)')
        
        ax.set_xlabel('Bitrate (kbps)', fontsize=13, fontweight='bold')
        ax.set_ylabel('PSNR (dB)', fontsize=13, fontweight='bold')
        ax.set_title(f'PSNR vs Bitrate - {dataset_name.capitalize()}', fontsize=15, fontweight='bold')
        ax.legend(fontsize=10, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        path = os.path.join(output_folder, 'PSNR_vs_bitrate.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ PSNR graph saved")
    
    # ========== SSIM GRAPH ==========
    if 'SSIM' in df.columns and not df['SSIM'].isna().all():
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for codec in df['Codec'].unique():
            data = df[df['Codec'] == codec].dropna(subset=['SSIM'])
            data = data.sort_values('Bitrate (kbps)') 
            if not data.empty:
                ax.plot(data['Bitrate (kbps)'], data['SSIM'], 
                       marker='o', label=f'{codec}', linewidth=2.5, markersize=10)
        
        threshold_colors = {'Minimal': 'green', 'Low': 'orange'}
        for label, threshold in SSIM_THRESHOLDS.items():
            ax.axhline(y=threshold, color=threshold_colors[label], linestyle='--', 
                      linewidth=1.5, alpha=0.7, label=f'{label} Degradation ({threshold})')
        
        ax.set_xlabel('Bitrate (kbps)', fontsize=13, fontweight='bold')
        ax.set_ylabel('SSIM', fontsize=13, fontweight='bold')
        ax.set_title(f'SSIM vs Bitrate - {dataset_name.capitalize()}', fontsize=15, fontweight='bold')
        ax.legend(fontsize=10, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        path = os.path.join(output_folder, 'SSIM_vs_bitrate.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ SSIM graph saved")
    
    # ========== VMAF GRAPH ==========
    if 'VMAF' in df.columns and not df['VMAF'].isna().all():
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for codec in df['Codec'].unique():
            data = df[df['Codec'] == codec].dropna(subset=['VMAF'])
            data = data.sort_values('Bitrate (kbps)') 
            if not data.empty:
                ax.plot(data['Bitrate (kbps)'], data['VMAF'], 
                       marker='o', label=f'{codec}', linewidth=2.5, markersize=10)
        
        threshold_colors = {'Excellent': 'green', 'Good': 'yellowgreen', 'Fair': 'orange'}
        for label, threshold in VMAF_THRESHOLDS.items():
            ax.axhline(y=threshold, color=threshold_colors[label], linestyle='--', 
                      linewidth=1.5, alpha=0.7, label=f'{label} ({threshold})')
        
        ax.set_xlabel('Bitrate (kbps)', fontsize=13, fontweight='bold')
        ax.set_ylabel('VMAF Score', fontsize=13, fontweight='bold')
        ax.set_title(f'VMAF vs Bitrate - {dataset_name.capitalize()}', fontsize=15, fontweight='bold')
        ax.legend(fontsize=10, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        path = os.path.join(output_folder, 'VMAF_vs_bitrate.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ VMAF graph saved")
    
    # ========== COMPRESSION EFFICIENCY ==========
    if 'PSNR (dB)' in df.columns and not df['PSNR (dB)'].isna().all():
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for codec in df['Codec'].unique():
            data = df[df['Codec'] == codec].dropna(subset=['PSNR (dB)'])
            data = data.sort_values('File Size (MB)') 
            if not data.empty:
                ax.plot(data['File Size (MB)'], data['PSNR (dB)'], 
                       marker='o', label=f'{codec}', linewidth=2.5, markersize=10)
        
        ax.set_xlabel('File Size (MB)', fontsize=13, fontweight='bold')
        ax.set_ylabel('PSNR (dB)', fontsize=13, fontweight='bold')
        ax.set_title(f'Compression Efficiency - {dataset_name.capitalize()}', fontsize=15, fontweight='bold')
        ax.legend(fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        path = os.path.join(output_folder, 'compression_efficiency.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Compression efficiency graph saved")

def generate_comparison_summary(all_results_df):
    """Generate comparison summary between datasets."""
    summary_path = os.path.join(BASE_RESULTS_FOLDER, 'comparison_summary.csv')
    
    # Calculate averages per dataset and codec
    summary = all_results_df.groupby(['Dataset', 'Codec']).agg({
        'PSNR (dB)': 'mean',
        'SSIM': 'mean',
        'VMAF': 'mean',
        'File Size (MB)': 'mean'
    }).round(2)
    
    summary.to_csv(summary_path)
    
    print(f"\n  ✓ Comparison summary saved: {summary_path}")
    print("\n" + "="*70)
    print("  DATASET COMPARISON SUMMARY")
    print("  (Average values across all bitrates)")
    print("="*70)
    print(summary.to_string())
    
    return summary

# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_dataset(dataset_name):
    """Complete processing pipeline for a single dataset."""
    print("\n" + "="*70)
    print(f"  PROCESSING: {dataset_name.upper()}")
    print("="*70)
    
    # Create directories
    output_folder, _, graph_folder, temp_folder = create_directories(dataset_name)
    
    # [1/6] Load images
    print(f"\n[1/6] Loading images from {dataset_name}/...")
    images = get_image_list(dataset_name)
    if not images:
        print(f"  ✗ No images found in {dataset_name}/ folder")
        return None
    print(f"  ✓ Found {len(images)} images")
    
    # [2/6] Prepare sequence
    print(f"\n[2/6] Preparing image sequence...")
    input_pattern = prepare_image_sequence(images, temp_folder)
    print(f"  ✓ Sequence prepared in {temp_folder}/")
    
    # [3/6] Create reference
    print(f"\n[3/6] Creating lossless reference video...")
    reference_video = create_reference_video(input_pattern, output_folder)
    if not reference_video:
        print(f"  ✗ Failed to create reference video")
        return None
    ref_size = get_file_size_mb(reference_video)
    print(f"  ✓ Reference created: {ref_size:.1f} MB")
    
    # [4/6] Encode videos
    print(f"\n[4/6] Encoding {len(CODECS) * len(BITRATES)} videos (parallel)...")
    encoded_videos = encode_all_videos_parallel(input_pattern, output_folder, dataset_name)
    if not encoded_videos:
        print(f"  ✗ No videos were encoded successfully")
        return None
    print(f"  ✓ Successfully encoded {len(encoded_videos)} videos")
    
    # [5/6] Evaluate quality
    print(f"\n[5/6] Evaluating quality metrics...")
    results_df = evaluate_videos(encoded_videos, reference_video, dataset_name)
    
    csv_path = os.path.join(BASE_RESULTS_FOLDER, f'{dataset_name}_quality_metrics.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"  ✓ Results saved: {csv_path}")
    
    # [6/6] Generate graphs
    print(f"\n[6/6] Generating graphs with quality thresholds...")
    generate_graphs_with_thresholds(results_df, dataset_name, graph_folder)
    
    print(f"\n✓✓✓ {dataset_name.upper()} PROCESSING COMPLETE ✓✓✓")
    
    return results_df

def main():
    """Main execution function."""
    start_time = time.time()
    
    print("\n" + "="*70)
    print("  LIGHT FIELD VIDEO ENCODING - FINAL VERSION")
    print("  Processing flowers and cards datasets")
    print("="*70)
    print(f"  Datasets: {', '.join(DATASETS)}")
    print(f"  Codecs: {', '.join(CODECS.keys())}")
    print(f"  Bitrates: {BITRATES} kbps")
    print(f"  Preset: {PRESET}")
    print(f"  Frame Rate: {FRAMERATE} fps")
    print("="*70)
    
    # Process both datasets
    all_results = []
    for dataset in DATASETS:
        result_df = process_dataset(dataset)
        if result_df is not None:
            all_results.append(result_df)
    
    # Generate comparison
    if len(all_results) > 0:
        print("\n" + "="*70)
        print("  GENERATING COMPARISON SUMMARY")
        print("="*70)
        combined_df = pd.concat(all_results, ignore_index=True)
        generate_comparison_summary(combined_df)
    
    # Final summary
    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print("  ✓✓✓ ALL PROCESSING COMPLETE ✓✓✓")
    print("="*70)
    print(f"  Total processing time: {elapsed/60:.1f} minutes")
    print(f"\n  Output Structure:")
    print(f"    output/")
    for dataset in DATASETS:
        print(f"      {dataset}/")
        print(f"        Videos:  {BASE_OUTPUT_FOLDER}/{dataset}/")
        print(f"        Results: {BASE_RESULTS_FOLDER}/{dataset}_quality_metrics.csv")
        print(f"        Graphs:  {BASE_GRAPHS_FOLDER}/{dataset}/")
    print(f"\n      Comparison: {BASE_RESULTS_FOLDER}/comparison_summary.csv")
    print("="*70)
    print("\n  Thank you for using Light Field Video Encoder!")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
