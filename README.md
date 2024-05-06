# Evaluation of DxGPT for <u>common and rare diseases</u> across multiple closed & open models

### Intro to this repo

Welcome to our repository dedicated to the evaluation of [DxGPT](https://dxgpt.app/) across various AI models, both for common and rare diseases. 

This project aims to explore the capabilities and limitations of different AI models in diagnosing diseases through synthetic and real-world datasets. Our comprehensive analysis includes closed models like GPT-4, Claude3 and open models like Llama2, Mistral and Cohere Command R +, providing insights into their diagnostic accuracy and potential applications in healthcare.

In this repository, you will find detailed evaluations, comparisons, and insights derived from multiple AI models. Our goal is to contribute to the advancement of AI in healthcare, specifically in improving diagnostic processes and outcomes for a wide range of diseases. We encourage collaboration, discussion, and feedback to further enhance the understanding and development of AI-driven diagnostics.

Stay tuned for updates and findings as we delve deeper into the world of AI and healthcare.

### Summary plots

![All Models Comparison](./allmodels_plot.png)

![Synthetic Dataset](./synthetic_plot.png)

![RAMEDIS Dataset](./RAMEDIS_plot.png)

![PUMCH Dataset](./PUMCH_plot.png)


### IPynb Dashboard with comparison of multiple models:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/foundation29org/dxgpt_testing/blob/main/dashboard.ipynb)

## File naming convention

The naming convention of the files in this repository is systematic and provides quick insights into the contents and purpose of each file. Understanding the naming structure will help you navigate and utilize the data effectively.

### Structure

Each file name is composed of four main parts:

1. **Evaluation data prefix**: All files related to model evaluation scores begin with `scores_`. This prefix is a clear indicator that the file contains data from the evaluation 
process. `diagnoses_` prefix is used for the files that contain the actual diagnoses from each test run, same naming convention as the scores files. `synthetic_*` prefix is used for the synthetic datasets.

2. Additionally, the dataset name is included to provide context. Example datasets include:
    - `(empty)` is a gpt4 synthetic dataset
    - `claude` is a claude 2 synthetic dataset
    - `medisearch` is a medisearch synthetic dataset
    - `RAMEDIS` is the RAMEDIS dataset from RareBench 
    - `PUMCH_ADM` is the PUMCH dataset from RareBench
    - `URG_Torre_Dic_200` is our proprietary dataset from common diseases in urgency care.

3. This is followed by the version of the dataset used for the evaluation (`(empty)` is v1, `v2` is the second version of the dataset).

4. **Model identifier**: Following the prefix, the name includes an identifier for the AI model used during the evaluation. Some of the possible model identifiers are:
   - `_gpt4_0613`: Data evaluated using the GPT-4 model checkpoint 0613.
   - `_llama`: Data evaluated using the LLaMA model.
   - `_c3`: Data evaluated using the Claude 3 model.
   - `_mistral`: Data evaluated using the Mistral model.
   - `_geminipro15`: Data evaluated using the Gemini Pro 1.5 model.

### Modifiers

In addition to the main parts, file names may include modifiers that provide further context about the evaluation:

- `_improved`: Indicates that the file contains data from an evaluation using an improved version of the prompt.
- `_rare_only_prompt`: Specifies that the evaluation prompt was a test focused exclusively on rare diseases.

### Examples

- `scores_v2_gpt4_0613.csv`: Evaluation scores from the second version of the dataset using the GPT-4 model checkpoint 0613.
- `scores_medisearch_v2_gpt4turbo1106.csv`: Evaluation scores from the medisearch synthetic dataset using the GPT-4 model turbo checkpoint 1106.
- `scores_URG_Torre_Dic_200_improved_c3sonnet.csv`: Evaluation scores from the urgency care dataset from December using the Claude 3 Sonnet model with an improved prompt.
- `scores_RAMEDIS_cohere_cplus.csv`: Evaluation scores from the RAMEDIS dataset using the Cohere Command R + model.
- `scores_PUMCH_ADM_mistralmoe.csv`: Evaluation scores from the PUMCH dataset using the Mistral MoE 8x7B model.

This structured approach to file naming ensures that each file is easily identifiable and that its contents are self-explanatory based on the name alone.

### Link to the DxGPT free web app:
[![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://dxgpt.app/)

# Evaluation of DxGPT live model accuracy for <u>rare diseases</u>   diagnoses
This repository contains all the code, data, and results for the evaluation of [DxGPT](https://github.com/foundation29org/Dx29_client_gpt)'s diagnostic accuracy on synthetic rare disease cases. The paper "Evaluation of DxGPT Accuracy for Rare Diseases Diagnoses" describes the methodology and findings of this analysis in detail. The goal of open sourcing this content is to provide full transparency on the evaluation process and enable further research to build on this work.

## Summary

[This paper](https://foundation29.sharepoint.com/:w:/s/Fundacion29-Share/Edy1Cl9pjdRLicmopJgPCeoBlwPpwjQ-Po07vLb-ZVXIWQ?e=HNViOk) evaluates DxGPT, a [web platform](https://dxgpt.app/) designed to accelerate the diagnosis of rare diseases. The platform uses GPT-4 to provide diagnostic suggestions based on a brief clinical description. The evaluation utilized 200 synthetic patient cases, derived from three models: GPT-4, Claude2, and MediSearch.

These 200 cases were selected from a curated list from Orphanet Prevalency List, a list of diseases that are the most common in rare diseases.

## Key features

- **Accelerated diagnosis**: Targets rare diseases that often face long diagnosis delays (average of 5-6 years).
- **Input**: Takes a brief clinical description.
- **Output**: Provides a ranked list of potential diagnoses.

## Evaluation metrics

- **Strict Accuracy (P1)**: Top suggestion matches the ground truth.
- **Top-5 Accuracy (P1+P5)**: Ground truth appears within the top 5 suggestions.

## Results

- 67.5% Strict Accuracy (for GPT-4 cases)
- 57% Strict Accuracy (for MediSearch cases)
- 88.5% Top-5 Accuracy (for GPT-4 cases)
- 83.5% Top-5 Accuracy (for MediSearch cases)

## Conclusions

The results are promising but require further validation on real clinical data and against human expert diagnoses.

## Future work

- Examine model performance per disease type.
- Investigate qualitative errors.
- Compare model performance to clinician baselines.

## Potential impact

With further rigorous evaluation, DxGPT shows potential to significantly assist doctors in diagnosing rare diseases faster, thus leading to improved patient outcomes.
