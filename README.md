# Galaxy Morphology Classification System
A hierarchical deep learning pipeline for automated galaxy morphology classification following the Hubble sequence taxonomy.

## In progress, This section relates to work done in first phase (semester) of development
Main work in baseline section

## Problem Statement
Modern astronomical surveys capture millions of galaxy images requiring morphological classification. Manual expert classification is time-intensive and doesn't scale. This system automates galaxy classification while maintaining accuracy comparable to domain experts.
## Dataset

- Source: EFIGI Catalogue (4,458 professionally-labeled images)
- Classes: 9 fine-grained morphological types across 4 coarse categories
- Split: 70% train (3,120), 10% validation (445), 20% test (891)
- Challenge: Severe class imbalance (rarest class: 0.9% of data)

## Architecture
### Three Hierarchical Approaches Implemented:

- Flat Baseline: Single classifier head (71.6% macro F1)
- Classifier Per Level: Separate heads for coarse/fine levels with probability weighting (72.8% macro F1)
- Classifier Per Node: Individual classifiers per parent node with conditional probability combination (74.9% macro F1)

## All models use:

- ResNet-18 backbone for feature extraction (ImageNet pre-trained, fine-tuned)
- Cross-entropy loss with balanced class weights
- Adam optimizer (lr=0.001) with ReduceLROnPlateau scheduling
- Data augmentation: random rotation (360°), horizontal/vertical flips

## Results
| Model | Fine F1 (Macro) | Coarse F1 (Macro) | Rarest Class F1 | 
| ----- | --------------- | ----------------- | --------------- |
| Flat | 71% | N/a | 46% |
| Per Level | 74% | 85% | 55% |
| Per Node | 75% | 85% | 66% |

## Key Innovation
The per-node architecture addresses class imbalance by creating specialized classifiers for each subtree, preventing overwhelming by majority classes. This approach improved rarest class accuracy by 29 percentage points.
## Technologies
PyTorch • scikit-learn • NumPy • Pandas • PIL • Google Colab
