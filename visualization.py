import wfdb
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Rectangle

def extract_annotated_segment(record_name, annotation_ext='q1c', directory='qtdb/1.0.0'):
    """
    Extract only the ECG segment between first and last annotations.
    
    Parameters:
    -----------
    record_name : str
        Name of the record to process
    annotation_ext : str
        Extension for the annotation file
    directory : str
        Directory containing the record files
        
    Returns:
    --------
    signals_annotated : ndarray
        The extracted signal data
    fields : dict
        Signal metadata
    ann_adjusted : Annotation object
        Adjusted annotations relative to the extracted segment
    """
    # Load the record
    signals, fields = wfdb.rdsamp(record_name, pn_dir=directory)
    
    # Load annotations
    ann = wfdb.rdann(record_name, annotation_ext, pn_dir=directory)
    
    # Find the first and last annotation sample points
    first_ann_sample = ann.sample[0]
    last_ann_sample = ann.sample[-1]
    
    print(f"First annotation at sample: {first_ann_sample}")
    print(f"Last annotation at sample: {last_ann_sample}")
    print(f"Extracted segment length: {last_ann_sample - first_ann_sample + 1} samples")
    print(f"Extraction timespan: {(last_ann_sample - first_ann_sample + 1) / fields['fs']:.2f} seconds")
    
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
        record_name=record_name,
        extension=annotation_ext,
        sample=adjusted_samples,
        symbol=adjusted_symbols,
        aux_note=None,  # We're skipping aux_note for simplicity
        chan=ann.chan,
        fs=ann.fs,
        label_store=None,  # We're skipping label_store for simplicity
        description=f"Adjusted annotations for {record_name}"
    )
    
    return signals_annotated, fields, ann_adjusted

def plot_extracted_segment(signals_annotated, fields, ann_adjusted, time_window=None, beat_centered=False):
    """
    Plot the extracted ECG segment with improved visualization of PQRST segments.
    
    Parameters:
    -----------
    signals_annotated : ndarray
        The extracted signal data
    fields : dict
        Signal metadata
    ann_adjusted : Annotation object
        Adjusted annotations
    time_window : tuple or None
        Optional (start_time, end_time) in seconds to plot a specific window
    beat_centered : bool
        If True and time_window is None, centers the plot on a single heartbeat
    """
    # Create a new time array starting from 0 for the annotated region
    fs = fields['fs']  # Sampling frequency
    time_annotated = np.arange(signals_annotated.shape[0]) / fs
    
    # If we want to center on a beat and no specific time window is provided
    if beat_centered and time_window is None:
        # Find R peaks (usually marked as 'N' in annotations)
        r_peak_indices = [i for i, symbol in enumerate(ann_adjusted.symbol) if symbol == 'N']
        
        if len(r_peak_indices) >= 2:
            # Get a complete heartbeat (from one R peak to just before the next)
            middle_r_index = len(r_peak_indices) // 2  # Choose a middle R peak
            r_peak_time = ann_adjusted.sample[r_peak_indices[middle_r_index]] / fs
            next_r_peak_time = ann_adjusted.sample[r_peak_indices[middle_r_index + 1]] / fs
            
            # Add some margin before and after
            margin = (next_r_peak_time - r_peak_time) * 0.5
            time_window = (r_peak_time - margin, next_r_peak_time + margin)
    
    # Default: show 2 seconds if no window specified and not beat-centered
    if time_window is None:
        if len(time_annotated) > 2 * fs:
            time_window = (0, 2.0)  # Show first 2 seconds
        else:
            time_window = (0, time_annotated[-1])  # Show all available data
    
    # Plot the signals
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot each signal for the annotated region with better visibility
    for i in range(signals_annotated.shape[1]):
        ax.plot(time_annotated, signals_annotated[:, i], label=fields['sig_name'][i], 
                linewidth=2.0, color='black', zorder=10)  # Higher zorder to keep signal on top
    
    # Dictionary for annotation symbols and their descriptions
    symbol_info = {
        'p': {'color': '#2ca02c', 'name': 'P wave', 'description': 'Atrial depolarization'},
        'N': {'color': '#d62728', 'name': 'R peak', 'description': 'Ventricular depolarization peak'},
        't': {'color': '#1f77b4', 'name': 'T wave', 'description': 'Ventricular repolarization'},
    }
    
    # Find segments in time
    segments = {}
    # Initialize segment times
    segment_times = {'P': [], 'QRS': [], 'T': []}
    
    # Extract segment boundaries
    for i in range(len(ann_adjusted.sample) - 1):
        current_symbol = ann_adjusted.symbol[i]
        next_symbol = ann_adjusted.symbol[i + 1]
        
        current_time = ann_adjusted.sample[i] / fs
        next_time = ann_adjusted.sample[i + 1] / fs
        
        # P wave: from 'p' to next annotation
        if current_symbol == 'p':
            segment_times['P'].append((current_time, next_time))
        
        # QRS complex: from '(' to ')'
        if current_symbol == '(' and next_symbol == 'N':
            # Find the closing parenthesis
            for j in range(i + 1, len(ann_adjusted.sample)):
                if ann_adjusted.symbol[j] == ')':
                    qrs_end = ann_adjusted.sample[j] / fs
                    segment_times['QRS'].append((current_time, qrs_end))
                    break
        
        # T wave: from 't' to next annotation
        if current_symbol == 't':
            segment_times['T'].append((current_time, next_time))
    
    # Add colored background regions for each segment with improved visibility
    segment_colors = {
        'P': '#90EE90',  # Brighter green
        'QRS': '#FFB6C1',  # Brighter pink
        'T': '#ADD8E6'   # Brighter blue
    }
    
    segment_borders = {
        'P': '#228B22',  # Darker green border
        'QRS': '#B22222',  # Darker red border
        'T': '#1E90FF'   # Darker blue border
    }
    
    # Plot the colored segments
    y_min, y_max = ax.get_ylim()
    height = y_max - y_min
    
    # Add slight vertical margin to make segments stand out better
    segment_margin = height * 0.05
    
    legend_handles = []
    
    # Create a legend entry for each segment type
    for segment_name, time_pairs in segment_times.items():
        if time_pairs:  # Only if we have segments of this type
            for start_time, end_time in time_pairs:
                if (start_time >= time_window[0] and start_time <= time_window[1]) or \
                   (end_time >= time_window[0] and end_time <= time_window[1]):
                    # Actual visible segment within window
                    visible_start = max(start_time, time_window[0])
                    visible_end = min(end_time, time_window[1])
                    
                    # Add a small gap between segments (5% of segment width or 0.01s, whichever is smaller)
                    gap = min((visible_end - visible_start) * 0.05, 0.01)
                    
                    # Create rectangle patch with border
                    rect = Rectangle((visible_start + gap/2, y_min + segment_margin), 
                                    (visible_end - visible_start) - gap, 
                                    height - 2*segment_margin, 
                                    facecolor=segment_colors[segment_name], 
                                    alpha=0.5,  # Increased opacity
                                    edgecolor=segment_borders[segment_name],
                                    linewidth=1,
                                    zorder=1)  # Keep segments behind the ECG trace
                    ax.add_patch(rect)
                    
                    # Add a subtle label in the middle of the segment if it's wide enough
                    segment_width = visible_end - visible_start
                    if segment_width > 0.1:  # Only label segments wider than 0.1s
                        ax.text(visible_start + segment_width/2, 
                               y_min + segment_margin*2,
                               segment_name, 
                               fontsize=8, 
                               color=segment_borders[segment_name],
                               ha='center', 
                               va='bottom',
                               zorder=3,
                               bbox=dict(facecolor='white', alpha=0.7, pad=0.1, edgecolor='none'))
            
            # Add to legend (only once per segment type)
            legend_handles.append(plt.Rectangle((0, 0), 1, 1, 
                                              color=segment_colors[segment_name],
                                              alpha=0.5, 
                                              label=f'{segment_name} Segment'))
    
    # Plot annotations more visibly with staggered labels
    # Group nearby annotations to avoid label overlap
    visible_annotations = []
    for i, (sample, symbol) in enumerate(zip(ann_adjusted.sample, ann_adjusted.symbol)):
        adjusted_time = sample / fs
        if time_window[0] <= adjusted_time <= time_window[1] and symbol in symbol_info:
            visible_annotations.append((adjusted_time, symbol, i))
    
    # Sort by time
    visible_annotations.sort(key=lambda x: x[0])
    
    # Calculate minimum time gap for label staggering
    min_gap = (time_window[1] - time_window[0]) / 20  # Minimum gap as fraction of visible window
    
    # Track label positions to avoid overlap
    last_label_end = {}  # Track the end position of the last label for each vertical position
    
    # Define vertical positions for staggering (as fractions of the plot height)
    # More positions = less chance of overlap
    vertical_positions = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    
    for i, (adjusted_time, symbol, orig_idx) in enumerate(visible_annotations):
        # Draw the vertical line at the annotation point
        ax.axvline(x=adjusted_time, 
                  color=symbol_info[symbol]['color'], 
                  linestyle='-', 
                  linewidth=1.5, 
                  alpha=0.8)
        
        # Find a suitable vertical position for this label
        label_text = symbol_info[symbol]['name']
        
        # Estimate text width in data units (approximate)
        text_width = len(label_text) * min_gap * 0.2
        
        # Find a position where this label won't overlap with others
        chosen_pos = None
        for pos in vertical_positions:
            # Check if this position is available
            if pos not in last_label_end or adjusted_time - text_width/2 > last_label_end[pos]:
                chosen_pos = pos
                break
        
        # If all positions are taken at this time point, use the last one
        if chosen_pos is None:
            chosen_pos = vertical_positions[-1]
        
        # Update the end position for this vertical level
        last_label_end[chosen_pos] = adjusted_time + text_width/2
        
        # Place label at the chosen position
        label = ax.text(adjusted_time, 
               y_min + height * chosen_pos, 
               label_text, 
               color=symbol_info[symbol]['color'], 
               fontweight='bold',
               ha='center', 
               fontsize=8,
               bbox=dict(facecolor='white', alpha=0.9, edgecolor=symbol_info[symbol]['color'], 
                        boxstyle='round,pad=0.2'))
        
        # Draw a connecting line from the annotation to the label
        ax.plot([adjusted_time, adjusted_time], 
               [y_min + height * chosen_pos * 0.95, y_max * 0.99 if symbol in ['p', 't'] else y_min * 0.99], 
               color=symbol_info[symbol]['color'], 
               linestyle=':', 
               linewidth=0.8,
               alpha=0.6)
    
    # Add the segment patches to the legend
    for handle in legend_handles:
        ax.add_artist(plt.legend(handles=[handle], loc='upper left', framealpha=0.9, fontsize=9))
    
    # Set the x-axis limits to the specified time window
    ax.set_xlim(time_window)
    
    # Create separate axes for the legend to avoid overcrowding the main plot
    legend_ax = fig.add_axes([0.125, -0.05, 0.75, 0.1], frameon=False)  # [left, bottom, width, height]
    legend_ax.axis('off')  # Hide the axis
    
    # Add a comprehensive legend to the separate axes
    legend_elements = []
    
    # ECG signal line
    legend_elements.append(plt.Line2D([0], [0], color='black', linewidth=2, label='ECG Signal'))
    
    # Annotation markers
    for symbol, info in symbol_info.items():
        legend_elements.append(plt.Line2D([0], [0], marker='|', color=info['color'], 
                                        markersize=15, linewidth=0, 
                                        label=f"{info['name']} ({info['description']})"))
    
    # Segment patches
    for segment_name in segment_colors.keys():
        if segment_name in segment_times and segment_times[segment_name]:
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, 
                                             color=segment_colors[segment_name],
                                             alpha=0.5, 
                                             label=f'{segment_name} Wave/Complex'))
    
    # Create the legend with 2 columns
    legend = legend_ax.legend(handles=legend_elements, loc='center', ncol=3, fontsize=9)
    
    # Add plot details
    ax.set_title(f'ECG Record: {ann_adjusted.record_name} (Focused View)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Amplitude (mV)', fontsize=12)
    
    # Only show signal name legend if there are multiple signals
    if signals_annotated.shape[1] > 1:
        signal_legend = ax.legend(loc='upper right', fontsize=9)
    
    # Lighter grid to avoid interference with ECG visualization
    ax.grid(True, alpha=0.2, linestyle=':')
    
    # Add additional information about the time window
    info_text = f"Time window: {time_window[0]:.2f}s to {time_window[1]:.2f}s (duration: {time_window[1]-time_window[0]:.2f}s)"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, ha='left', va='top', 
           fontsize=9, bbox=dict(facecolor='white', alpha=0.8, pad=0.5, edgecolor='lightgray'))
    
    return fig

# Main execution
if __name__ == "__main__":
    # Record name
    record_name = 'sel100'
    
    # Extract the annotated segment
    signals_annotated, fields, ann_adjusted = extract_annotated_segment(
        record_name, 
        annotation_ext='q1c', 
        directory='qtdb/1.0.0',
    )
    
    # Calculate better visualization windows
    
    # Find all R peaks
    r_peak_indices = [i for i, symbol in enumerate(ann_adjusted.symbol) if symbol == 'N']
    r_peak_times = [ann_adjusted.sample[i] / fields['fs'] for i in r_peak_indices]
    
    if len(r_peak_times) >= 2:
        # Calculate the average RR interval
        rr_intervals = [r_peak_times[i+1] - r_peak_times[i] for i in range(len(r_peak_times)-1)]
        avg_rr_interval = sum(rr_intervals) / len(rr_intervals)
        
        # Option 1: Very focused view showing a single beat in detail
        # Center on a middle R peak and show 60% of an RR interval before and 80% after
        middle_r_index = len(r_peak_times) // 2
        focused_start = r_peak_times[middle_r_index] - (avg_rr_interval * 0.6)
        focused_end = r_peak_times[middle_r_index] + (avg_rr_interval * 0.8)
        
        # Option 2: Show exactly 2 complete beats
        # Find two consecutive beats in the middle
        middle_r_index = len(r_peak_times) // 2
        if middle_r_index + 1 < len(r_peak_times):
            two_beat_start = r_peak_times[middle_r_index-1] - (avg_rr_interval * 0.2)
            two_beat_end = r_peak_times[middle_r_index+1] + (avg_rr_interval * 0.2)
        else:
            two_beat_start = r_peak_times[0] - (avg_rr_interval * 0.2)
            two_beat_end = r_peak_times[1] + (avg_rr_interval * 0.2)
        
        # Option 3: Show approximately 3 to 4 beats
        wider_start = max(0, r_peak_times[0])
        wider_end = min(r_peak_times[-1], wider_start + avg_rr_interval * 4)
    else:
        # Fallback if we don't have enough R peaks
        signal_duration = len(signals_annotated) / fields['fs']
        focused_start, focused_end = 0, min(0.8, signal_duration)
        two_beat_start, two_beat_end = 0, min(1.6, signal_duration)
        wider_start, wider_end = 0, min(3.0, signal_duration)
    
    # Plot options:
    
    # Option 1: Plot a single beat with detailed view
    plot_extracted_segment(signals_annotated, fields, ann_adjusted, time_window=(focused_start, focused_end))
    plt.savefig('ecg_single_beat_detail.png', dpi=300, bbox_inches='tight')
    
    # Option 2: Plot two complete beats
    plot_extracted_segment(signals_annotated, fields, ann_adjusted, time_window=(two_beat_start, two_beat_end))
    plt.savefig('ecg_two_beats.png', dpi=300, bbox_inches='tight')
    
    # Option 3: Plot multiple beats for rhythm assessment
    plot_extracted_segment(signals_annotated, fields, ann_adjusted, time_window=(wider_start, wider_end))
    plt.savefig('ecg_multiple_beats.png', dpi=300, bbox_inches='tight')
    
    # Display the plots
    plt.show()