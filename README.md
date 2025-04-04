# PQRST-HMM: ECG Segmentation using Hidden Markov Models

This repository contains an implementation of a Hidden Markov Model (HMM) approach for automatic segmentation of electrocardiogram (ECG) signals into their component waves (P, QRS, T) and baseline regions.

## Background

ECG signals capture the electrical activity of the heart and contain several distinct components:
- P wave: Representing atrial depolarization
- QRS complex: Representing ventricular depolarization
- T wave: Representing ventricular repolarization

Automated segmentation of these components is crucial for various medical applications including arrhythmia detection, heart rate monitoring, and general cardiac health assessment.

## Requirements

The project uses Python 3.12 with the following dependencies:
- numpy
- scipy
- pandas
- scikit-learn
- matplotlib
- wfdb
- hmmlearn
- seaborn
- jupyter

You can set up the environment using conda:

```bash
conda env create -f environment.yml
conda activate PQRST-HMM
```

## Dataset

This implementation uses the QT Database from PhysioNet, which contains annotated ECG signals. The annotations include markers for P waves, QRS complexes, and T waves.

The dataset is not included in this repository and should be downloaded separately and placed in a directory named `qt-database`.

## Implementation Overview

The project consists of several Python modules:

- `load_data.py`: Functions for loading ECG data from the QT Database
- `plot.py`: Functions for visualizing ECG signals and annotations
- `train.py`: Functions for feature extraction, HMM training, and segmentation
- `evaluation.py`: Functions for evaluating the segmentation performance
- `visualization.py`: Scripts for visualizing ECG signals and segmentation results

The main methodology includes:
1. Feature extraction from ECG signals (signal values, derivatives, windowed statistics)
2. HMM training with domain knowledge initialization
3. Segmentation using the trained HMM
4. Mapping HMM states to ECG waves
5. Evaluation against manual annotations

## Current Performance

As shown in the terminal output, the current implementation has significant limitations:

```
Validation Metrics Summary:
QRS Precision: 0.6613
QRS Recall: 0.8948
QRS F1-Score: 0.7025
P Wave Precision: 0.1103
P Wave Recall: 0.8498
P Wave F1-Score: 0.1872
T Wave Precision: 0.2795
T Wave Recall: 0.4058
T Wave F1-Score: 0.2901
Macro Precision: 0.3504
Macro Recall: 0.7168
Macro F1-Score: 0.3932
```

The model performs reasonably well for QRS complex detection (F1-Score: 0.7025) but struggles with P waves (F1-Score: 0.1872) and T waves (F1-Score: 0.2901).

## Limitations and Future Work

Major issues identified in the evaluation:
1. Poor discrimination between P waves and baseline
2. Significant misclassification of T waves as P waves (54.59%)
3. Class imbalance issues with QRS waves making up only 3.50% of samples

Potential improvements:
1. Enhanced feature extraction to better discriminate between wave types
2. Experiment with different HMM parameters or state transition models
3. Address class imbalance through sampling techniques
4. Explore supervised approaches as suggested in the referenced paper

## Alternative Approaches

As mentioned in the output, the implementation in this repository did not produce meaningful results. For better performance, consider the approach described in:

[Emission Modelling for Supervised ECG Segmentation using Finite Differences](https://www.researchgate.net/publication/226679477_Emission_Modelling_for_Supervised_ECG_Segmentation_using_Finite_Differences)

## Usage

To run the main evaluation:

```bash
python evaluation.py
```

To visualize example ECG segments:

```bash
python visualization.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The QT Database from PhysioNet for providing annotated ECG recordings
- The researchers behind the referenced paper for providing methodological insights