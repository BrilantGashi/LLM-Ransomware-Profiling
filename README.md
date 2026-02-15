# LLM-Based Ransomware Negotiation Profiling

## Abstract
This project implements a computational framework for the automated analysis of ransomware negotiations using Large Language Models (LLMs). The system processes negotiation transcripts through a multi-agent pipeline to apply Speech Act Theory for linguistic classification, derive psychological profiles of threat actors, and extract tactical intelligence. The architecture employs an "Agentic Consensus" mechanism to synthesize and validate outputs from multiple models, producing structured data for forensic analysis.

## Project Overview

### Research Objectives
1.  **Automated Behavioral Analysis**: To develop a scalable pipeline for processing unstructured negotiation logs into structured behavioral datasets.
2.  **Linguistic Classification**: To categorize communicative intents within cyber-extortion contexts based on Speech Act Theory.
3.  **Multi-Model Validation**: To mitigate LLM inconsistencies by utilizing an ensemble approach with an adjudicator agent to synthesize results.
4.  **Forensic Profiling**: To extract correlations between linguistic patterns, psychological traits, and negotiation tactics.

### Methodology
The system processes data through the following stages:
1.  **Ingestion**: Raw negotiation transcripts are normalized and anonymized.
2.  **Parallel Inference**: Multiple LLMs (e.g., Qwen, Phi-4, Llama-3) independently analyze the data across three dimensions: Speech Acts, Psychological Profiling, and Tactical Extraction.
3.  **Agentic Adjudication**: A dedicated consensus agent evaluates the independent model outputs, resolving discrepancies based on linguistic consistency to produce a validated JSON record.
4.  **Aggregation**: Validated records are consolidated into structured datasets (CSV) for statistical analysis and temporal modeling.

## System Architecture

The project is structured as a modular Python application (`generative_ai_project`), designed for deployment on high-performance computing clusters (UniBS GPUStack).

### Directory Structure

```text
generative_ai_project/
├── config/                     # Configuration Management
│   ├── model_config.yaml       # Global system settings (models, parameters, paths)
│   ├── prompt_templates.yaml   # Base prompt engineering templates
│   └── agentic_consensus_prompts.yaml # Specialized prompts for the Consensus Agent
│
├── src/                        # Source Code
│   ├── llm/
│   │   └── unibs_client.py     # OpenAI-compatible API client for the UniBS cluster
│   │
│   ├── analysis/
│   │   ├── agentic_consensus.py # Logic for LLM-based adjudication and synthesis
│   │   ├── aggregator.py       # Data transformation pipeline (JSON -> CSV)
│   │   ├── consensus.py        # Legacy mathematical consensus implementation
│   │   └── visualizer.py       # Plotting utilities for data analysis
│   │
│   ├── handlers/               # Utility Handlers
│   │   └── error_handler.py    # Error management and retry logic
│   │
│   └── utils/                  # Helper Utilities
│       └── data_loader.py      # Dataset ingestion and normalization
│
├── data/                       # Data Storage (Git-ignored)
│   ├── raw/                    # Original Ransomchats dataset
│   ├── outputs/                # Raw inference results from individual models
│   ├── consensus_agentic/      # Validated "Gold Standard" outputs
│   └── processed/              # Final CSV datasets for analysis
│
├── run_pipeline.py             # Entry point for the primary inference pipeline
├── run_agentic_consensus.py    # Entry point for the consensus adjudication phase
├── update_database.py          # Utility to sync with external data sources
└── requirements.txt            # Project dependencies manifest
```
Key Components
--------------

1. Multi-Model Inference Engine
The system utilizes an ensemble of Large Language Models to analyze each negotiation. Configuration is managed via 'config/model_config.yaml', allowing for dynamic model selection and parameter tuning. The 'LLMClient' class handles API communication with exponential backoff for reliability.

2. Agentic Consensus Module (src/analysis/agentic_consensus.py)
This module implements the logic for synthesizing multi-model outputs.
- Input: Independent JSON analyses from the ensemble models for a single chat.
- Process: An LLM (configured as 'consensus.agentic.active_model') acts as an adjudicator, evaluating the inputs against criteria defined in 'agentic_consensus_prompts.yaml'. It resolves conflicts and merges data into a single coherent output.
- Resilience: The module is designed to function with partial inputs (1 or 2 models) and logs reliability metrics.

3. Data Aggregator (src/analysis/aggregator.py)
Post-consensus, this module transforms the validated JSON documents into tabular formats suitable for quantitative research. It generates:
- Negotiation Dataset: Chat-level metrics (financial demands, duration, outcome).
- Speech Act Dataset: Message-level linguistic classification.
- Temporal Matrices: Time-series data representing the evolution of negotiation tactics.

Installation & Setup
--------------------

Prerequisites:
- Python 3.9+
- Access to the UniBS GPUStack API (or compatible OpenAI-API endpoint).
- Environment variable GPUSTACK_API_KEY set.

Step-by-Step Installation:

1. Clone the Repository
   git clone https://github.com/BrilantGashi/LLM-Ransomware-Profiling.git
   cd LLM-Ransomware-Profiling/generative_ai_project

2. Install Dependencies
   The project relies on a strictly defined set of libraries. Use the provided requirements file to install the environment:
   pip install -r requirements.txt

3. Configure API Access
   The project uses environment variables for security. A template is provided.
   Duplicate the example file to create your local configuration:
   cp .env.example .env
   
   Open '.env' and insert your personal GPUSTACK_API_KEY.

Execution Workflow
------------------

1. Data Synchronization
   Downloads and prepares the raw dataset.
   python update_database.py

2. Primary Inference (Multi-Model Analysis)
   Runs the ensemble models on the raw dataset to generate initial JSON outputs.
   python run_pipeline.py

3. Consensus Adjudication
   Synthesizes the raw outputs into validated "Gold Standard" data using the Agentic Consensus module.
   python run_agentic_consensus.py --task all

4. Data Aggregation
   Generates the final CSV datasets for analysis from the consensus data.
   (Note: Ensure aggregator is configured to read from data/consensus_agentic)
   python src/analysis/aggregator.py

Technical Dependencies
----------------------
- OpenAI SDK: For standardized API interaction with LLMs.
- Pandas: For data manipulation and aggregation.
- PyYAML: For configuration management.
- Matplotlib/Seaborn: For data visualization (via visualizer.py).

Author
------
Brilant Gashi
Department of Information Engineering
University of Brescia
2024-2025
