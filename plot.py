import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_extracted_segment(signals_annotated, fields, ann_adjusted, time_window=None, title=None):
    """
    Plot the extracted ECG segment with annotations.
    
    Parameters:
    -----------
    signals_annotated : numpy.ndarray
        The extracted ECG signals
    fields : dict
        Metadata about the signal
    ann_adjusted : wfdb.Annotation
        Adjusted annotations for the segment
    time_window : tuple, optional
        (start_time, end_time) in seconds to focus the plot on a specific timeframe
    title : str, optional
        Title for the plot
    """
    # Create a time axis based on sampling frequency
    fs = fields['fs']
    time = np.arange(len(signals_annotated)) / fs
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Plot the ECG signal (first channel)
    ax.plot(time, signals_annotated[:, 0], 'b', linewidth=1.5)
    
    # Set time window if provided
    if time_window:
        start_time, end_time = time_window
        ax.set_xlim(start_time, end_time)
        
        # Calculate sample indices for the visible portion
        start_idx = max(0, int(start_time * fs))
        end_idx = min(len(signals_annotated), int(end_time * fs))
        
        # Adjust y-axis limits to focus on the visible signal
        visible_signal = signals_annotated[start_idx:end_idx, 0]
        if len(visible_signal) > 0:
            min_val = np.min(visible_signal)
            max_val = np.max(visible_signal)
            y_margin = (max_val - min_val) * 0.2
            ax.set_ylim(min_val - y_margin, max_val + y_margin)
    
    # Colors for annotation markers and wave segments
    marker_colors = {
        'p': 'green',    # P wave
        'N': 'red',      # R peak (QRS complex)
        't': 'purple',   # T wave
        '(': 'black',    # Start marker
        ')': 'black'     # End marker
    }
    
    segment_colors = {
        'p': 'lightgreen',    # P wave 
        'N': 'lightcoral',    # QRS complex
        't': 'plum'           # T wave
    }
    
    wave_labels = {
        'p': 'P wave',
        'N': 'QRS complex',
        't': 'T wave'
    }
    
    # Find and highlight wave segments
    i = 0
    while i < len(ann_adjusted.symbol):
        wave_type = None
        start_sample = None
        end_sample = None
        
        # Check for standard triplets: '(', wave_type, ')'
        if (i < len(ann_adjusted.symbol) - 2 and 
            ann_adjusted.symbol[i] == '(' and 
            ann_adjusted.symbol[i+1] in ['p', 'N', 't'] and 
            ann_adjusted.symbol[i+2] == ')'):
            
            wave_type = ann_adjusted.symbol[i+1]
            start_sample = ann_adjusted.sample[i]
            end_sample = ann_adjusted.sample[i+2]
            increment = 3  # Skip entire triplet
            
        # Check for wave markers without starting '('
        elif (i < len(ann_adjusted.symbol) - 1 and 
              ann_adjusted.symbol[i] in ['p', 'N', 't'] and 
              ann_adjusted.symbol[i+1] == ')'):
            
            wave_type = ann_adjusted.symbol[i]
            start_sample = ann_adjusted.sample[i]  # Use middle mark as start
            end_sample = ann_adjusted.sample[i+1]
            increment = 2  # Skip wave and closing parenthesis
            
        # Check for wave markers without ending ')'
        elif (i > 0 and 
              ann_adjusted.symbol[i-1] == '(' and 
              ann_adjusted.symbol[i] in ['p', 'N', 't'] and 
              (i == len(ann_adjusted.symbol) - 1 or ann_adjusted.symbol[i+1] != ')')):
            
            wave_type = ann_adjusted.symbol[i]
            start_sample = ann_adjusted.sample[i-1]
            end_sample = ann_adjusted.sample[i]  # Use middle mark as end
            increment = 1  # Skip just the wave marker
            
        # Check for lone wave markers (neither starting '(' nor ending ')')
        elif ann_adjusted.symbol[i] in ['p', 'N', 't']:
            wave_type = ann_adjusted.symbol[i]
            start_sample = ann_adjusted.sample[i]  # Use the mark itself as both start and end
            end_sample = ann_adjusted.sample[i]
            
            # Special handling for T waves without parentheses - extend slightly
            if wave_type == 't':
                # Extend the end time slightly to make the T wave visible
                if i < len(ann_adjusted.symbol) - 1:
                    # If there's a next annotation, go halfway toward it
                    end_sample = ann_adjusted.sample[i] + (ann_adjusted.sample[i+1] - ann_adjusted.sample[i]) // 2
                else:
                    # Otherwise extend by a fixed amount (e.g., 50 samples)
                    end_sample = ann_adjusted.sample[i] + 50
            
            increment = 1  # Skip just the wave marker
        
        else:
            i += 1
            continue
        
        # Process this wave segment
        if wave_type:
            # Convert to time
            t_start = start_sample / fs
            t_end = end_sample / fs
            
            # Skip if outside time window
            if time_window and (t_end < start_time or t_start > end_time):
                i += increment
                continue
            
            # Get y-range of this segment
            start_idx = max(0, start_sample)
            end_idx = min(len(signals_annotated), end_sample + 1)
            segment_signal = signals_annotated[start_idx:end_idx, 0]
            
            if len(segment_signal) > 0:
                min_val = np.min(segment_signal)
                max_val = np.max(segment_signal)
                height = max_val - min_val
                
                # Add some margin
                y_margin = height * 0.3
                
                # Add background rectangle
                rect = Rectangle((t_start, min_val - y_margin), 
                                max(t_end - t_start, 0.01),  # Ensure minimum width
                                height + 2*y_margin,
                                facecolor=segment_colors.get(wave_type, 'lightgray'), 
                                alpha=0.3)
                ax.add_patch(rect)
                
                # Add label at the center of the segment
                center_time = (t_start + t_end) / 2
                label_pos = max_val + y_margin / 2
                ax.text(center_time, label_pos, wave_labels.get(wave_type, ''), 
                        horizontalalignment='center', fontsize=9)
        
        i += increment
    
    # Plot annotation markers
    for sample, symbol in zip(ann_adjusted.sample, ann_adjusted.symbol):
        ann_time = sample / fs
        
        # Skip if outside time window
        if time_window and (ann_time < start_time or ann_time > end_time):
            continue
        
        # Get signal value at this point
        y_val = signals_annotated[sample, 0]
        
        # Plot marker and label
        color = marker_colors.get(symbol, 'gray')
        ax.plot(ann_time, y_val, 'o', color=color, markersize=6)
        ax.text(ann_time, y_val, symbol, fontsize=9, 
                verticalalignment='bottom', horizontalalignment='center')
    
    # Add labels and title
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude (mV)')
    
    if time_window:
        ax.set_title(f'ECG Segment with PQRST Annotations ({start_time:.2f}s - {end_time:.2f}s)')
    else:
        ax.set_title('ECG Segment with PQRST Annotations')
    
    # Create legend for wave segments
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=segment_colors['p'], alpha=0.3, label='P wave'),
        plt.Rectangle((0, 0), 1, 1, facecolor=segment_colors['N'], alpha=0.3, label='QRS complex'),
        plt.Rectangle((0, 0), 1, 1, facecolor=segment_colors['t'], alpha=0.3, label='T wave')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if title:
        plt.title(title)
        plt.savefig(f'{title}.png')
        plt.close()
    else:
        plt.show()