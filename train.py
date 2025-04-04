import wfdb
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Rectangle
import glob
from collections import defaultdict
import random
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from hmmlearn import hmm
from matplotlib.patches import Patch

# Function to load all records from a directory
from load_data import load_all_ecg_data, extract_annotated_segment

# Function to extract features from ECG signal
def extract_features(signals, fields):
    """
    Extract features from ECG signal for HMM training and prediction.
    
    Parameters:
    signals (numpy.ndarray): ECG signals
    fields (dict): Metadata for the signals
    
    Returns:
    numpy.ndarray: Feature matrix
    """
    # For simplicity, we'll use the first lead only
    signal = signals[:, 0]
    
    # Compute derivative (slope)
    derivative = np.diff(signal, prepend=signal[0])
    
    # Compute second derivative
    second_derivative = np.diff(derivative, prepend=derivative[0])
    
    # Create windowed statistical features
    window_size = int(0.1 * fields['fs'])  # 100ms window
    
    # Initialize arrays
    mean_feature = np.zeros_like(signal)
    var_feature = np.zeros_like(signal)
    
    # Compute windowed features
    for i in range(len(signal)):
        start = max(0, i - window_size // 2)
        end = min(len(signal), i + window_size // 2)
        window = signal[start:end]
        mean_feature[i] = np.mean(window)
        var_feature[i] = np.var(window)
    
    # Normalize features
    signal_norm = zscore(signal)
    derivative_norm = zscore(derivative)
    second_derivative_norm = zscore(second_derivative)
    mean_feature_norm = zscore(mean_feature)
    var_feature_norm = zscore(var_feature)
    
    # Combine features
    features = np.column_stack((signal_norm, derivative_norm, second_derivative_norm, 
                              mean_feature_norm, var_feature_norm))
    
    return features

# Function to initialize HMM with domain knowledge about ECG
def initialize_hmm(n_states=4, n_features=5):
    """
    Initialize HMM with domain knowledge about ECG waves.
    
    Parameters:
    n_states (int): Number of states (baseline, P, QRS, T)
    n_features (int): Number of features
    
    Returns:
    hmm.GaussianHMM: Initialized HMM
    """
    # Create HMM
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", 
                           algorithm="viterbi", random_state=42, init_params="")
    
    # Initial state distribution (start with baseline)
    model.startprob_ = np.array([0.7, 0.1, 0.1, 0.1])
    
    # Transition matrix (baseline -> P -> QRS -> T -> baseline)
    model.transmat_ = np.array([
        [0.7, 0.3, 0.0, 0.0],  # From baseline
        [0.0, 0.2, 0.8, 0.0],  # From P wave
        [0.0, 0.0, 0.2, 0.8],  # From QRS complex
        [0.3, 0.0, 0.0, 0.7]   # From T wave
    ])
    
    # Initialize means based on domain knowledge
    # State 0: Baseline - near zero, low derivatives
    # State 1: P wave - small positive, small derivatives
    # State 2: QRS complex - large positive/negative, high derivatives
    # State 3: T wave - medium positive, medium derivatives
    model.means_ = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],  # Baseline
        [0.5, 0.2, 0.0, 0.5, 0.2],  # P wave
        [2.0, 5.0, 5.0, 2.0, 5.0],  # QRS complex
        [1.0, 0.5, 0.0, 1.0, 0.5]   # T wave
    ])
    
    # Initialize covariances as identity matrices
    model.covars_ = np.array([np.eye(n_features) for _ in range(n_states)])
    
    return model

# Function to train HMM on ECG data
def train_hmm(features_list, model=None, n_states=4):
    """
    Train an HMM on ECG data.
    
    Parameters:
    features_list (list): List of feature matrices
    model (hmm.GaussianHMM, optional): Pre-initialized HMM
    n_states (int): Number of states
    
    Returns:
    hmm.GaussianHMM: Trained HMM
    """
    # Prepare data
    X = np.vstack(features_list)
    lengths = [len(x) for x in features_list]
    
    # Initialize model if not provided
    if model is None:
        model = initialize_hmm(n_states, X.shape[1])
    
    # Train the model
    model.fit(X, lengths=lengths)
    
    return model

# Function to segment ECG using trained HMM
def segment_ecg(model, features):
    """
    Segment ECG using a trained HMM.
    
    Parameters:
    model (hmm.GaussianHMM): Trained HMM
    features (numpy.ndarray): Feature matrix
    
    Returns:
    numpy.ndarray: Predicted state sequence
    """
    # Predict states
    states = model.predict(features)
    
    return states

# Function to map HMM states to ECG waves using annotations
def map_states_to_ecg_waves(states, annotations, signals_length, fs):
    """
    Map HMM states to ECG waves using annotations.
    
    Parameters:
    states (numpy.ndarray): HMM state sequence
    annotations (wfdb.Annotation): ECG annotations
    signals_length (int): Length of the signal
    fs (float): Sampling frequency
    
    Returns:
    dict: Mapping from HMM state to ECG wave
    """
    # # Print unique annotation symbols to understand what's available
    # unique_symbols = set(annotations.symbol)
    # print(f"Unique annotation symbols: {unique_symbols}")
    
    # Get R peak samples (QRS complex)
    r_peaks = [i for i, symbol in enumerate(annotations.symbol) if symbol == 'N']
    r_peak_samples = [annotations.sample[i] for i in r_peaks]
    
    # Count state occurrences at R peaks (QRS complex)
    qrs_counts = np.zeros(np.max(states) + 1)
    for sample in r_peak_samples:
        if 0 <= sample < signals_length:
            qrs_counts[states[sample]] += 1
    
    # Identify QRS state (highest count at R peaks)
    qrs_state = np.argmax(qrs_counts)
    
    # Identify P wave state (state that often precedes QRS)
    p_counts = np.zeros(np.max(states) + 1)
    for sample in r_peak_samples:
        if sample > int(0.2 * fs):  # Look ~200ms before R peak
            p_window = states[sample-int(0.2*fs):sample]
            for s in range(np.max(states) + 1):
                p_counts[s] += np.sum(p_window == s)
    
    # Exclude QRS state from P wave candidates
    p_counts[qrs_state] = 0
    p_state = np.argmax(p_counts)
    
    # Identify T wave state (state that often follows QRS)
    t_counts = np.zeros(np.max(states) + 1)
    for sample in r_peak_samples:
        if sample + int(0.4 * fs) < signals_length:  # Look ~400ms after R peak
            t_window = states[sample:sample+int(0.4*fs)]
            for s in range(np.max(states) + 1):
                t_counts[s] += np.sum(t_window == s)
    
    # Exclude QRS and P states from T wave candidates
    t_counts[qrs_state] = 0
    t_counts[p_state] = 0
    t_state = np.argmax(t_counts)
    
    # Identify baseline state (remaining state with highest count)
    state_counts = np.bincount(states)
    state_counts[qrs_state] = 0
    state_counts[p_state] = 0
    state_counts[t_state] = 0
    baseline_state = np.argmax(state_counts)
    
    # Create mapping
    mapping = {
        'baseline': baseline_state,
        'p': p_state,
        'qrs': qrs_state,
        't': t_state
    }
    
    # Print mapping
    # print("State mapping:")
    # for wave, state in mapping.items():
    #     print(f"{wave} -> State {state}")
    
    return mapping

# Main function
def main():
    # Set random seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    
    # Load all ECG data from the qt-database
    print("Loading ECG data...")
    ecg_data = load_all_ecg_data()
    
    if not ecg_data:
        print("No ECG data found")
        return
    
    # Lists to store features for training
    features_list = []
    
    # Select records with q1c annotations
    record_list = [r for r in ecg_data.keys() if 'q1c' in ecg_data[r]['annotations']]
    
    if not record_list:
        print("No records with q1c annotations found")
        return
    
    print(f"Found {len(record_list)} records with q1c annotations")
    
    # Split into training and testing records
    train_records, test_records = train_test_split(record_list, test_size=0.2, random_state=seed)
    
    print(f"Using {len(train_records)} records for training and {len(test_records)} for testing")
    
    # Process training records
    for record in train_records:
        # print(f"\nProcessing record for training: {record}")
        
        # Extract annotated segment
        segment_data = extract_annotated_segment(ecg_data[record], 'q1c')
        
        if segment_data:
            signals = segment_data['signals']
            fields = segment_data['fields']
            
            # Extract features
            features = extract_features(signals, fields)
            
            # Add to training data
            features_list.append(features)
    
    if not features_list:
        print("No valid features extracted for training")
        return
    
    print("\nTraining HMM...")
    # Train HMM with domain knowledge initialization
    model = train_hmm(features_list)
    
    print("HMM training complete!")
    
    # Process test records
    print("\nSegmenting test records:")
    
    for record in test_records:
        print(f"\nSegmenting record: {record}")
        
        # Extract annotated segment
        segment_data = extract_annotated_segment(ecg_data[record], 'q1c')
        
        if segment_data:
            signals = segment_data['signals']
            fields = segment_data['fields']
            annotations = segment_data['annotations']
            
            # Extract features
            features = extract_features(signals, fields)
            
            # Segment using HMM
            states = segment_ecg(model, features)
            
            # Map states to ECG waves using annotations
            mapping = map_states_to_ecg_waves(states, annotations, len(signals), fields['fs'])
            
            # Evaluate segmentation (print some statistics)
            # Calculate duration of each segment
            total_samples = len(states)
            segment_counts = {wave: np.sum(states == state) for wave, state in mapping.items()}
            segment_percentages = {wave: (count / total_samples) * 100 
                                  for wave, count in segment_counts.items()}
            
            print("\nSegmentation Statistics:")
            for wave, percentage in segment_percentages.items():
                print(f"{wave.capitalize()} segments: {percentage:.2f}% of signal")

# Entry point
if __name__ == "__main__":
    main()