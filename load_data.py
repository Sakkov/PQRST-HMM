import wfdb
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Rectangle
import glob
from collections import defaultdict
import random

from plot import plot_extracted_segment

# Create a dictionary to store all loaded ECG data
ecg_data = {}

# Function to load all records from a directory
def load_all_ecg_data(database_dir='qt-database'):
    # Find all record files (those with .dat extension) in the database directory
    record_files = glob.glob(f"{database_dir}/*.dat")
    
    # Extract record names without extension
    record_names = [os.path.splitext(os.path.basename(file))[0] for file in record_files]
    
    print(f"Found {len(record_names)} records in {database_dir}")
    
    # Dictionary to store all loaded data
    all_data = {}
    
    # Load each record and its annotations
    for record_name in record_names:
        full_record_path = f"{database_dir}/{record_name}"
        
        try:
            # Load the record
            signals, fields = wfdb.rdsamp(full_record_path, pn_dir=None)
            
            # Only look for q1c annotation files
            ann_file = f"{full_record_path}.q1c"
            
            record_annotations = {}
            
            # Only load q1c annotations if they exist
            if os.path.exists(ann_file):
                try:
                    ann = wfdb.rdann(full_record_path, 'q1c', pn_dir=None)
                    record_annotations['q1c'] = ann
                    
                    print(f"Loaded {record_name} with q1c annotation: {len(ann.sample)} annotations")
                    
                    # Store the data
                    all_data[record_name] = {
                        'signals': signals,
                        'fields': fields,
                        'annotations': record_annotations
                    }
                except Exception as e:
                    print(f"Failed to load q1c annotation for {record_name}: {e}")
            else:
                print(f"No q1c annotation found for {record_name}")
            
        except Exception as e:
            print(f"Failed to load record {record_name}: {e}")
    
    return all_data

# Function to extract annotated segment from a record
def extract_annotated_segment(record_data, annotation_ext='q1c'):
    signals = record_data['signals']
    fields = record_data['fields']
    
    if annotation_ext not in record_data['annotations']:
        print(f"Annotation {annotation_ext} not found for this record")
        return None
    
    ann = record_data['annotations'][annotation_ext]
    
    # Find the first and last annotation sample points
    first_ann_sample = ann.sample[0]
    last_ann_sample = ann.sample[-1]
    
    # print(f"First annotation at sample: {first_ann_sample}")
    # print(f"Last annotation at sample: {last_ann_sample}")
    # print(f"Extracted segment length: {last_ann_sample - first_ann_sample + 1} samples")
    # print(f"Extraction timespan: {(last_ann_sample - first_ann_sample + 1) / fields['fs']:.2f} seconds")
    
    # Extract only the annotated region of the signal
    signals_annotated = signals[first_ann_sample:last_ann_sample+1, :]
    
    # Manually filter and adjust annotations
    adjusted_samples = []
    adjusted_symbols = []
    
    # Process each annotation and keep only those in our range
    for i in range(len(ann.sample)):
        if first_ann_sample <= ann.sample[i] <= last_ann_sample:
            adjusted_samples.append(ann.sample[i] - first_ann_sample)
            adjusted_symbols.append(ann.symbol[i])
    
    # Convert to numpy arrays
    adjusted_samples = np.array(adjusted_samples)
    adjusted_symbols = np.array(adjusted_symbols)
    
    # Create the adjusted annotation object
    ann_adjusted = wfdb.Annotation(
        record_name=ann.record_name,
        extension=annotation_ext,
        sample=adjusted_samples,
        symbol=adjusted_symbols,
        aux_note=None,  # We're skipping aux_note for simplicity
        chan=ann.chan,
        fs=ann.fs,
        label_store=None,  # We're skipping label_store for simplicity
        description=f"Adjusted annotations for {ann.record_name}"
    )
    
    # print(f"Adjusted annotations: {len(adjusted_samples)} found")
    
    return {
        'signals': signals_annotated,
        'fields': fields,
        'annotations': ann_adjusted
    }

# Function to analyze beat statistics
def analyze_beats(record_data, segment_data=None, annotation_ext='q1c'):
    if segment_data is None:
        signals = record_data['signals']
        fields = record_data['fields']
        ann = record_data['annotations'][annotation_ext]
    else:
        signals = segment_data['signals']
        fields = segment_data['fields']
        ann = segment_data['annotations']
    
    # Find all R peaks (normal beats)
    r_peak_indices = [i for i, symbol in enumerate(ann.symbol) if symbol == 'N']
    r_peak_samples = [ann.sample[i] for i in r_peak_indices]
    r_peak_times = [sample / fields['fs'] for sample in r_peak_samples]
    
    # Calculate RR intervals
    rr_intervals = [r_peak_times[i+1] - r_peak_times[i] for i in range(len(r_peak_times)-1)]
    
    # Calculate statistics
    stats = {
        'num_beats': len(r_peak_indices),
        'avg_rr_interval': np.mean(rr_intervals) if rr_intervals else None,
        'std_rr_interval': np.std(rr_intervals) if rr_intervals else None,
        'min_rr_interval': np.min(rr_intervals) if rr_intervals else None,
        'max_rr_interval': np.max(rr_intervals) if rr_intervals else None,
        'heart_rate': 60 / np.mean(rr_intervals) if rr_intervals else None
    }
    
    # Count annotation types
    annotation_counts = defaultdict(int)
    for symbol in ann.symbol:
        annotation_counts[symbol] += 1
    
    return {
        'r_peaks': r_peak_samples,
        'r_peak_times': r_peak_times,
        'rr_intervals': rr_intervals,
        'stats': stats,
        'annotation_counts': annotation_counts
    }

# Function to plot a representative beat
def plot_representative_beat(segment_data, beat_analysis):
    signals = segment_data['signals']
    fields = segment_data['fields']
    ann = segment_data['annotations']
    
    if not beat_analysis['r_peak_times']:
        print("No R peaks found to plot")
        return
    
    # Center on a middle R peak and show 60% of an RR interval before and 80% after
    middle_r_index = len(beat_analysis['r_peak_times']) // 2
    avg_rr_interval = beat_analysis['stats']['avg_rr_interval']
    
    focused_start = beat_analysis['r_peak_times'][middle_r_index] - (avg_rr_interval * 0.6)
    focused_end = beat_analysis['r_peak_times'][middle_r_index] + (avg_rr_interval * 0.8)
    
    # Plot a single beat with detailed view
    plot_extracted_segment(signals, fields, ann, time_window=(focused_start, focused_end))

# Main execution
if __name__ == "__main__":

    seed = 42
    random.seed(seed)

    # Load all ECG data from the qt-database
    ecg_data = load_all_ecg_data()
    
    # Demo with the random 10 records 
    if ecg_data:
        record_list = list(ecg_data.keys())
        total_records = len(record_list)
        
        if total_records == 0:
            print("No records with q1c annotations found")
        else:
            # Analyze up to 10 records, but don't attempt more than available
            for i in range(min(10, total_records)):
                # Choose a random record with q1c annotations
                chosen_i = random.randint(0, total_records - 1)
                record = record_list[chosen_i]
                
                print(f"\nAnalyzing record with q1c annotation: {record}")
                
                # Extract annotated segment
                segment_data = extract_annotated_segment(ecg_data[record], 'q1c')
                
                if segment_data:
                    # Analyze beats
                    beat_analysis = analyze_beats(None, segment_data)
                    
                    # Print some statistics
                    print("\nBeat Statistics:")
                    for key, value in beat_analysis['stats'].items():
                        if value is not None:
                            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
                    
                    # Plot a representative beat
                    plot_representative_beat(segment_data, beat_analysis)