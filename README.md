
# Evaluation of DxGPT Accuracy for Rare Diseases Diagnoses

## Summary

This paper evaluates DxGPT, a web platform designed to accelerate the diagnosis of rare diseases. The platform uses GPT-4 to provide diagnostic suggestions based on a brief clinical description. The evaluation utilized 200 synthetic patient cases, derived from three models: GPT-4, Claude2, and MediSearch.

## Key Features

- **Accelerated Diagnosis**: Targets rare diseases that often face long diagnosis delays (average of 5-6 years).
- **Input**: Takes a brief clinical description.
- **Output**: Provides a ranked list of potential diagnoses.

## Evaluation Metrics

- **Strict Accuracy (P1)**: Top suggestion matches the ground truth.
- **Top-5 Accuracy (P1+P5)**: Ground truth appears within the top 5 suggestions.

## Results

- 67.5% Strict Accuracy (for GPT-4 cases)
- 57% Strict Accuracy (for MediSearch cases)
- 88.5% Top-5 Accuracy (for GPT-4 cases)
- 83.5% Top-5 Accuracy (for MediSearch cases)

## Conclusions

The results are promising but require further validation on real clinical data and against human expert diagnoses.

## Future Work

- Examine model performance per disease type.
- Investigate qualitative errors.
- Compare model performance to clinician baselines.

## Potential Impact

With further rigorous evaluation, DxGPT shows potential to significantly assist doctors in diagnosing rare diseases faster, thus leading to improved patient outcomes.