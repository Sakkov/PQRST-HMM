import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib.lines import Line2D
import random

from load_data import load_all_ecg_data, extract_annotated_segment
from train import extract_features, segment_ecg, train_hmm, map_states_to_ecg_waves

def evaluate_segmentation_comprehensive(ecg_data, records, model):
    """
    Comprehensive evaluation of the HMM segmentation by considering the annotated
    wave boundaries and evaluating segmentation at those regions.
    
    Parameters:
    ecg_data (dict): Dictionary containing ECG data
    records (list): List of record names
    model (hmm.GaussianHMM): Trained HMM model
    
    Returns:
    dict: Evaluation metrics
    """
    # Initialize metrics
    metrics = {
        'qrs_precision': 0, 'qrs_recall': 0, 'qrs_f1': 0,
        'p_precision': 0, 'p_recall': 0, 'p_f1': 0,
        't_precision': 0, 't_recall': 0, 't_f1': 0,
        'macro_precision': 0, 'macro_recall': 0, 'macro_f1': 0,
        'confusion_matrix': np.zeros((4, 4))  # baseline, p, qrs, t
    }
    
    num_evaluated = 0
    
    # Process each record
    for record_idx, record in enumerate(records):
        # print(f"\nEvaluating record {record_idx+1}/{len(records)}: {record}")
        
        # Extract annotated segment
        segment_data = extract_annotated_segment(ecg_data[record], 'q1c')
        
        if not segment_data:
            continue
            
        signals = segment_data['signals']
        fields = segment_data['fields']
        annotations = segment_data['annotations']
        
        # Extract features
        features = extract_features(signals, fields)
        
        # Segment using HMM
        predicted_states = segment_ecg(model, features)
        
        # Map states to ECG waves using annotations
        mapping = map_states_to_ecg_waves(predicted_states, annotations, len(signals), fields['fs'])
        
        # Convert predicted states to waves
        predicted_waves = np.full(len(signals), 'baseline', dtype=object)
        for wave, state in mapping.items():
            predicted_waves[predicted_states == state] = wave
        
        # Create ground truth waves based on annotations
        # This is a simplified approach assuming we can create approximate boundaries
        ground_truth_waves = np.full(len(signals), 'baseline', dtype=object)
        
        # Parameter: window sizes around annotation points
        qrs_window = int(0.08 * fields['fs'])  # 80ms for QRS complex
        p_window = int(0.10 * fields['fs'])    # 100ms for P wave
        t_window = int(0.15 * fields['fs'])    # 150ms for T wave
        
        # Mark QRS regions around R peaks
        for idx in [i for i, symbol in enumerate(annotations.symbol) if symbol == 'N']:
            sample = annotations.sample[idx]
            if 0 <= sample < len(ground_truth_waves):
                start = max(0, sample - qrs_window // 2)
                end = min(len(ground_truth_waves), sample + qrs_window // 2)
                ground_truth_waves[start:end] = 'qrs'
        
        # Mark P wave regions before QRS complexes
        for idx in [i for i, symbol in enumerate(annotations.symbol) if symbol == 'p']:
            sample = annotations.sample[idx]
            if 0 <= sample < len(ground_truth_waves):
                start = max(0, sample - p_window // 2)
                end = min(len(ground_truth_waves), sample + p_window // 2)
                ground_truth_waves[start:end] = 'p'
        
        # Mark T wave regions after QRS complexes
        for idx in [i for i, symbol in enumerate(annotations.symbol) if symbol == 't']:
            sample = annotations.sample[idx]
            if 0 <= sample < len(ground_truth_waves):
                start = max(0, sample - t_window // 2)
                end = min(len(ground_truth_waves), sample + t_window // 2)
                ground_truth_waves[start:end] = 't'
        
        # Convert to numerical labels for metrics calculation
        wave_to_num = {'baseline': 0, 'p': 1, 'qrs': 2, 't': 3}
        predicted_waves_num = np.array([wave_to_num[w] for w in predicted_waves])
        ground_truth_waves_num = np.array([wave_to_num[w] for w in ground_truth_waves])
        
        # Update confusion matrix
        for i in range(4):  # For each true wave type
            for j in range(4):  # For each predicted wave type
                mask = (ground_truth_waves_num == i) & (predicted_waves_num == j)
                metrics['confusion_matrix'][i, j] += np.sum(mask)
        
        # Calculate per-wave metrics for this record
        record_metrics = {}
        for wave, wave_idx in [('p', 1), ('qrs', 2), ('t', 3)]:
            # True positives, false positives, false negatives
            true_pos = np.sum((ground_truth_waves_num == wave_idx) & (predicted_waves_num == wave_idx))
            false_pos = np.sum((ground_truth_waves_num != wave_idx) & (predicted_waves_num == wave_idx))
            false_neg = np.sum((ground_truth_waves_num == wave_idx) & (predicted_waves_num != wave_idx))
            
            # Calculate precision, recall, F1
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Store record metrics
            record_metrics[f'{wave}_precision'] = precision
            record_metrics[f'{wave}_recall'] = recall
            record_metrics[f'{wave}_f1'] = f1
            
            # Print record metrics
            # print(f"{wave.upper()} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Update overall metrics
        for metric in record_metrics:
            metrics[metric] += record_metrics[metric]
        
        num_evaluated += 1
    
    # Average the metrics across records
    if num_evaluated > 0:
        for metric in metrics:
            if metric != 'confusion_matrix':
                metrics[metric] /= num_evaluated
        
        # Calculate macro averages
        metrics['macro_precision'] = (metrics['p_precision'] + metrics['qrs_precision'] + metrics['t_precision']) / 3
        metrics['macro_recall'] = (metrics['p_recall'] + metrics['qrs_recall'] + metrics['t_recall']) / 3
        metrics['macro_f1'] = (metrics['p_f1'] + metrics['qrs_f1'] + metrics['t_f1']) / 3
    
    return metrics

def visualize_segmentation(signals, fields, predicted_states, mapping, annotations, record_name, ground_truth_waves=None):
    """
    Visualize the ECG segmentation results with comparison to ground truth if provided.
    
    Parameters:
    signals (numpy.ndarray): ECG signals
    fields (dict): Metadata for the signals
    predicted_states (numpy.ndarray): Predicted state sequence
    mapping (dict): Mapping from ECG wave to HMM state
    annotations (wfdb.Annotation): ECG annotations
    record_name (str): Name of the record
    ground_truth_waves (numpy.ndarray, optional): Ground truth wave labels
    """
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot ECG signal with annotations
    plt.subplot(211)
    plt.plot(signals[:, 0], color='black')
    
    # Add annotations
    for i, symbol in enumerate(annotations.symbol):
        sample = annotations.sample[i]
        if 0 <= sample < len(signals):
            if symbol == 'N':  # R peak
                plt.axvline(x=sample, color='red', linestyle='--', alpha=0.5)
                plt.text(sample, np.max(signals[:, 0]), 'R', color='red')
            elif symbol == 'p':  # P wave
                plt.axvline(x=sample, color='green', linestyle='--', alpha=0.5)
                plt.text(sample, np.max(signals[:, 0]), 'P', color='green')
            elif symbol == 't':  # T wave
                plt.axvline(x=sample, color='blue', linestyle='--', alpha=0.5)
                plt.text(sample, np.max(signals[:, 0]), 'T', color='blue')
    
    plt.title(f'ECG Signal with Annotations - {record_name}')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    # Plot segmentation
    plt.subplot(212)
    plt.plot(signals[:, 0], color='black')
    
    # Color the segments based on the predicted states
    colors = {'baseline': 'gray', 'p': 'green', 'qrs': 'red', 't': 'blue'}
    
    for wave, state in mapping.items():
        mask = (predicted_states == state)
        plt.fill_between(range(len(signals)), 
                         np.min(signals[:, 0]), 
                         np.max(signals[:, 0]), 
                         where=mask, 
                         color=colors[wave], 
                         alpha=0.3)
    
    # Create legend
    legend_elements = [Line2D([0], [0], color=color, lw=4, label=wave.upper())
                      for wave, color in colors.items()]
    plt.legend(handles=legend_elements, loc='best')
    
    plt.title(f'ECG Segmentation - {record_name}')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'segmentation_{record_name}.png')
    plt.close()

    # If ground truth is provided, create a comparison visualization
    if ground_truth_waves is not None:
        plt.figure(figsize=(15, 12))
        
        # Plot ECG signal
        plt.subplot(311)
        plt.plot(signals[:, 0], color='black')
        plt.title(f'ECG Signal - {record_name}')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        # Plot ground truth segmentation
        plt.subplot(312)
        plt.plot(signals[:, 0], color='black', alpha=0.5)
        
        for wave, color in colors.items():
            mask = (ground_truth_waves == wave)
            plt.fill_between(range(len(signals)), 
                            np.min(signals[:, 0]), 
                            np.max(signals[:, 0]), 
                            where=mask, 
                            color=color, 
                            alpha=0.3)
        
        plt.title(f'Ground Truth Segmentation - {record_name}')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        # Plot predicted segmentation
        plt.subplot(313)
        plt.plot(signals[:, 0], color='black', alpha=0.5)
        
        for wave, state in mapping.items():
            mask = (predicted_states == state)
            plt.fill_between(range(len(signals)), 
                            np.min(signals[:, 0]), 
                            np.max(signals[:, 0]), 
                            where=mask, 
                            color=colors[wave], 
                            alpha=0.3)
        
        plt.title(f'Predicted Segmentation - {record_name}')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'segmentation_comparison_{record_name}.png')
        plt.close()

def evaluate_hmm_segmentation():
    """
    Evaluate the performance of the HMM segmentation on validation data.
    """
    # Set random seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    
    # Load all ECG data from the qt-database
    print("Loading ECG data...")
    ecg_data = load_all_ecg_data()
    
    if not ecg_data:
        print("No ECG data found")
        return None, None
    
    # Select records with q1c annotations
    record_list = [r for r in ecg_data.keys() if 'q1c' in ecg_data[r]['annotations']]
    
    if not record_list:
        print("No records with q1c annotations found")
        return None, None
    
    print(f"Found {len(record_list)} records with q1c annotations")
    
    # Split into training, validation, and testing records
    train_records, val_records = train_test_split(record_list, test_size=0.3, random_state=seed)
    
    print(f"Using {len(train_records)} records for training and {len(val_records)} for validation")
    
    # Lists to store features for training
    features_list = []
    
    # Process training records
    for record in train_records:
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
        return None, None
    
    print("\nTraining HMM...")
    # Train HMM with domain knowledge initialization
    model = train_hmm(features_list)
    print("HMM training complete!")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_metrics = evaluate_segmentation_comprehensive(ecg_data, val_records, model)
    
    # Print validation metrics
    print("\nValidation Metrics Summary:")
    print(f"QRS Precision: {val_metrics['qrs_precision']:.4f}")
    print(f"QRS Recall: {val_metrics['qrs_recall']:.4f}")
    print(f"QRS F1-Score: {val_metrics['qrs_f1']:.4f}")
    print(f"P Wave Precision: {val_metrics['p_precision']:.4f}")
    print(f"P Wave Recall: {val_metrics['p_recall']:.4f}")
    print(f"P Wave F1-Score: {val_metrics['p_f1']:.4f}")
    print(f"T Wave Precision: {val_metrics['t_precision']:.4f}")
    print(f"T Wave Recall: {val_metrics['t_recall']:.4f}")
    print(f"T Wave F1-Score: {val_metrics['t_f1']:.4f}")
    print(f"Macro Precision: {val_metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {val_metrics['macro_recall']:.4f}")
    print(f"Macro F1-Score: {val_metrics['macro_f1']:.4f}")
    
    # Print confusion matrix
    print("\nConfusion Matrix (rows: true, columns: predicted):")
    wave_labels = ['baseline', 'p', 'qrs', 't']
    print("                " + " ".join([f"{w:10}" for w in wave_labels]))
    for i, row_label in enumerate(wave_labels):
        row_str = f"{row_label:10} | "
        for j in range(4):
            row_str += f"{val_metrics['confusion_matrix'][i, j]:10.0f} "
        print(row_str)
    
    # Visualize some examples
    print("\nVisualizing some examples...")
    num_visualize = min(5, len(val_records))  # Visualize at most 5 records
    for record in val_records[:num_visualize]:
        segment_data = extract_annotated_segment(ecg_data[record], 'q1c')
        
        if segment_data:
            signals = segment_data['signals']
            fields = segment_data['fields']
            annotations = segment_data['annotations']
            
            # Extract features
            features = extract_features(signals, fields)
            
            # Segment using HMM
            predicted_states = segment_ecg(model, features)
            
            # Map states to ECG waves using annotations
            mapping = map_states_to_ecg_waves(predicted_states, annotations, len(signals), fields['fs'])
            
            # Create ground truth waves based on annotations for visualization
            ground_truth_waves = np.full(len(signals), 'baseline', dtype=object)
            
            # Parameter: window sizes around annotation points
            qrs_window = int(0.08 * fields['fs'])  # 80ms for QRS complex
            p_window = int(0.10 * fields['fs'])    # 100ms for P wave
            t_window = int(0.15 * fields['fs'])    # 150ms for T wave
            
            # Mark QRS regions around R peaks
            for idx in [i for i, symbol in enumerate(annotations.symbol) if symbol == 'N']:
                sample = annotations.sample[idx]
                if 0 <= sample < len(ground_truth_waves):
                    start = max(0, sample - qrs_window // 2)
                    end = min(len(ground_truth_waves), sample + qrs_window // 2)
                    ground_truth_waves[start:end] = 'qrs'
            
            # Mark P wave regions
            for idx in [i for i, symbol in enumerate(annotations.symbol) if symbol == 'p']:
                sample = annotations.sample[idx]
                if 0 <= sample < len(ground_truth_waves):
                    start = max(0, sample - p_window // 2)
                    end = min(len(ground_truth_waves), sample + p_window // 2)
                    ground_truth_waves[start:end] = 'p'
            
            # Mark T wave regions
            for idx in [i for i, symbol in enumerate(annotations.symbol) if symbol == 't']:
                sample = annotations.sample[idx]
                if 0 <= sample < len(ground_truth_waves):
                    start = max(0, sample - t_window // 2)
                    end = min(len(ground_truth_waves), sample + t_window // 2)
                    ground_truth_waves[start:end] = 't'
            
            # Visualize with ground truth comparison
            visualize_segmentation(signals, fields, predicted_states, mapping, 
                                 annotations, record, ground_truth_waves)
            print(f"Saved visualization for record {record}")
    
    return model, val_metrics

def print_confusion_matrix_analysis(confusion_matrix):
    """
    Print a detailed analysis of the confusion matrix.
    
    Parameters:
    confusion_matrix (numpy.ndarray): Confusion matrix (4x4) for baseline, p, qrs, t
    """
    wave_labels = ['baseline', 'p', 'qrs', 't']
    
    print("\nConfusion Matrix Analysis:")
    
    # Calculate total samples for each true class
    class_totals = np.sum(confusion_matrix, axis=1)
    
    for i, true_label in enumerate(wave_labels):
        print(f"\nTrue {true_label.upper()} waves:")
        if class_totals[i] == 0:
            print(f"  No {true_label} samples in ground truth")
            continue
            
        for j, pred_label in enumerate(wave_labels):
            percentage = (confusion_matrix[i, j] / class_totals[i]) * 100 if class_totals[i] > 0 else 0
            print(f"  Classified as {pred_label.upper()}: {confusion_matrix[i, j]:.0f} samples ({percentage:.2f}%)")
        
        # Highlight major misclassifications
        if i > 0:  # Skip baseline
            misclass_indices = [j for j in range(4) if j != i]
            major_misclass = max(misclass_indices, key=lambda j: confusion_matrix[i, j])
            if confusion_matrix[i, major_misclass] > 0:
                percentage = (confusion_matrix[i, major_misclass] / class_totals[i]) * 100
                if percentage > 15:  # Arbitrary threshold for "significant" misclassification
                    print(f"  Note: Significant misclassification as {wave_labels[major_misclass].upper()} ({percentage:.2f}%)")

# Main entry point
if __name__ == "__main__":
    model, metrics = evaluate_hmm_segmentation()
    
    if metrics:
        # Additional analysis
        print("\n" + "="*50)
        print("DETAILED EVALUATION RESULTS")
        print("="*50)
        
        # Analyze confusion matrix
        print_confusion_matrix_analysis(metrics['confusion_matrix'])
        
        # Calculate per-wave accuracy
        total_samples = np.sum(metrics['confusion_matrix'])
        wave_labels = ['baseline', 'p', 'qrs', 't']
        
        print("\nPer-Wave Statistics:")
        for i, wave in enumerate(wave_labels):
            true_samples = np.sum(metrics['confusion_matrix'][i, :])
            pred_samples = np.sum(metrics['confusion_matrix'][:, i])
            correct = metrics['confusion_matrix'][i, i]
            
            if true_samples > 0:
                print(f"\n{wave.upper()} Wave:")
                print(f"  True samples: {true_samples:.0f} ({true_samples/total_samples*100:.2f}% of total)")
                print(f"  Predicted samples: {pred_samples:.0f} ({pred_samples/total_samples*100:.2f}% of total)")
                print(f"  Correctly classified: {correct:.0f} ({correct/true_samples*100:.2f}% of true {wave} samples)")
                
                if i > 0:  # Skip baseline for precision/recall/F1
                    wave_idx = wave  # p, qrs, t
                    print(f"  Precision: {metrics[f'{wave_idx}_precision']:.4f}")
                    print(f"  Recall: {metrics[f'{wave_idx}_recall']:.4f}")
                    print(f"  F1-Score: {metrics[f'{wave_idx}_f1']:.4f}")
        
        # Overall assessment
        print("\nOverall Assessment:")
        f1_scores = [metrics[f'{w}_f1'] for w in ['p', 'qrs', 't']]
        avg_f1 = sum(f1_scores) / len(f1_scores)
        
        if avg_f1 >= 0.8:
            performance = "Excellent"
        elif avg_f1 >= 0.7:
            performance = "Good"
        elif avg_f1 >= 0.6:
            performance = "Moderate"
        elif avg_f1 >= 0.5:
            performance = "Fair"
        else:
            performance = "Poor"
            
        print(f"  Overall performance: {performance} (Macro F1: {avg_f1:.4f})")
        
        # Recommendations based on results
        print("\nRecommendations:")
        
        # Find worst performing wave
        wave_f1 = [(w, metrics[f'{w}_f1']) for w in ['p', 'qrs', 't']]
        worst_wave, worst_f1 = min(wave_f1, key=lambda x: x[1])
        
        print(f"  1. Focus on improving {worst_wave.upper()} wave detection (F1: {worst_f1:.4f})")
        
        # Check for class imbalance
        class_percentages = [np.sum(metrics['confusion_matrix'][i, :]) / total_samples for i in range(4)]
        min_class = min(enumerate(class_percentages), key=lambda x: x[1])
        if min_class[1] < 0.1:  # Arbitrary threshold for imbalance
            print(f"  2. Address class imbalance: {wave_labels[min_class[0]].upper()} waves make up only {min_class[1]*100:.2f}% of samples")
        
        # Feature recommendations
        print("  3. Consider enhancing the feature extraction to better discriminate between wave types")
        
        # Model recommendations
        print("  4. Experiment with different HMM parameters or transitions to better model the cardiac cycle")