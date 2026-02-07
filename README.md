# LLM Ransomware Negotiation Profiling

Advanced psychological profiling and tactical extraction from ransomware negotiations using Large Language Models and Speech Act Theory.

**Bachelor's Thesis Project** - University of Brescia (2024-2025)

---

## 🎓 Academic Information

**Student**: Brilant Gashi  
**Institution**: University of Brescia (Università degli Studi di Brescia)  
**Degree Program**: Computer Science (Informatica)  
**Academic Year**: 2024-2025

### Supervisors

- **Prof. Federico Cerutti** -  Supervisor
- **Prof. Pietro Baroni** - Supervisor

**Department**: Department of Information Engineering (Dipartimento di Ingegneria dell'Informazione)

---

## 🎯 Thesis Overview

This bachelor's thesis investigates the application of Large Language Models (LLMs) for automated analysis of ransomware negotiation chats. The research leverages the **UniBS experimental LLM cluster** to process real-world negotiation data from the Ransomchats dataset, applying Speech Act Theory and multi-model ensemble techniques.

### Research Objectives

1. **Automated Negotiation Analysis**: Develop a scalable pipeline for processing ransomware negotiations
2. **Speech Act Classification**: Apply linguistic theory to categorize negotiation messages
3. **Psychological Profiling**: Extract behavioral patterns and negotiation tactics
4. **Multi-Model Validation**: Compare and consensus across 7 different LLMs
5. **Academic Contribution**: Advance cybersecurity research and negotiation analysis

### Key Research Questions

- Can LLMs accurately classify speech acts in ransomware negotiations?
- How do different models compare in understanding malicious communication?
- What consensus mechanisms provide the most reliable results?
- What psychological patterns emerge from automated analysis?


## 🌟 Features

### Core Capabilities

- ✅ **7 LLM Models**: qwen3, phi4-mini, phi4, llama3.2, gpt-oss, granite3.3, gemma3
- ✅ **UniBS Cluster Integration**: Production-ready API client (Handbook-compliant)
- ✅ **Multi-Task Pipeline**: Speech acts, psychological profiling, tactical extraction
- ✅ **Consensus Mechanisms**: Majority vote and weighted averaging
- ✅ **Robust Error Handling**: Exponential backoff with comprehensive logging
- ✅ **Few-Shot Learning**: Template-based prompt engineering
- ✅ **Dataset Management**: Automated updates from Ransomchats repository
- ✅ **Reasoning Capture**: Model thinking process extraction

---

## 📁 Project Structure

generative_ai_project/
├── config/ # Configuration files
│ ├── model_config.yaml # LLM parameters & cluster settings
│ ├── prompt_templates.yaml # Task prompts & instructions
│ ├── logging_config.yaml # Multi-level logging setup
│ └── few_shot_examples/ # Few-shot learning templates
│
├── src/ # Source code
│ ├── llm/
│ │ └── unibs_client.py # UniBS cluster API client
│ ├── handlers/
│ │ └── error_handler.py # Retry logic & error reporting
│ ├── utils/
│ │ ├── data_loader.py # Dataset loading utilities
│ │ └── debug_helper.py # Debugging tools
│ └── analysis/
│ ├── consensus.py # Multi-model consensus
│ ├── aggregator.py # Result aggregation
│ └── visualizer.py # Data visualization
│
├── data/ # Data storage
│ ├── raw/
│ │ ├── messages.json # Unified dataset
│ │ ├── DATA_MANIFEST.json # Dataset metadata
│ │ └── Ransomchats-main/ # Raw GitHub data
│ ├── outputs/ # Pipeline results
│ └── consensus/ # Cross-model validation
│
├── logs/ # Execution logs
│
├── run_pipeline.py # Main execution script
├── update_database.py # Dataset updater
└── requirements.txt # Python dependencies


## 🚀 Getting Started

### Prerequisites

- **Python 3.9+**
- **UniBS Network Access** (on-campus or via VPN)
- **UniBS GPUStack API Key** (provided by thesis supervisors)

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/BrilantGashi/LLM-Ransomware-Profiling.git
cd LLM-Ransomware-Profiling/generative_ai_project
2. Create Virtual Environment
bash
python3 -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
3. Install Dependencies
bash
pip install --upgrade pip
pip install -r requirements.txt
4. Configure Environment
bash
# Create .env file from template
cp .env.example .env

# Edit and add your API key
nano .env
.env content:

bash
GPUSTACK_API_KEY=your-api-key-here
GPUSTACK_BASE_URL=https://gpustack.ing.unibs.it/v1
⚠️ Security: Never commit .env to Git!

5. Download Dataset
bash
python update_database.py
🎮 Usage
Quick Test Run (5 chats)
bash
python run_pipeline.py
Full Dataset Processing
Edit run_pipeline.py line 385:

python
pipeline.run(max_chats=None)  # Process all chats
Then run:

bash
python run_pipeline.py
Configuration
Change Active Model
Edit config/model_config.yaml:

text
# Single model
active_model: "phi4-mini"

# Ensemble (multiple models)
ensemble_models:
  - phi4-mini
  - qwen3
  - llama3.2
Adjust LLM Parameters
text
llm_parameters:
  temperature: 0.6
  top_p: 0.95
  max_tokens: 1024
📊 Output Structure
Results are saved in data/outputs/:

text
data/outputs/
├── speech_act_analysis/
│   ├── phi4-mini/
│   │   └── [group_name]/
│   │       └── [chat_id].json
│   └── qwen3/
├── psychological_profiling/
└── tactical_extraction/
Consensus Results
When using ensemble mode:

text
data/consensus/
└── [group_name]/
    └── [chat_id]_consensus.json
🔬 Research Methodology
Pipeline Architecture
Data Ingestion: Load and clean ransomware negotiation chats

Prompt Engineering: Apply task-specific templates with few-shot examples

Multi-Model Inference: Process with 7 different LLMs

Result Validation: JSON parsing and schema validation

Consensus Generation: Cross-model agreement voting

Statistical Analysis: Aggregate results and extract patterns

Speech Act Theory
Classification based on Searle's taxonomy:

Assertives: Claims, statements, descriptions

Directives: Demands, requests, questions

Commissives: Promises, threats, offers

Expressives: Emotional expressions

Declarations: Status changes, confirmations

🧪 Testing
Test Logging Configuration
bash
python -m tests.test_logging
Verify API Connectivity
bash
cd src/utils
python debug_helper.py

📚 Documentation
Key Files
UniBS Cluster Handbook: Official API documentation

Configuration Guide: YAML configuration reference

API Reference: Source code documentation

Related Research
Ransomchats Dataset: github.com/Casualtek/Ransomchats

Speech Act Theory: Searle, J.R. (1969)

LLM Ensembles: Multi-model consensus techniques

🔒 Security & Ethics
✅ API Keys via Environment Variables (never committed)

✅ VPN-Only Access to UniBS cluster

✅ No Sensitive Data in public repository

👤 Author
Brilant Gashi
Computer Science Student
University of Brescia

🔗 GitHub: @BrilantGashi

🙏 Acknowledgments
Prof. Federico Cerutti, Pietro Baroni - Thesis supervision and guidance

UniBS IT Services - Access to experimental LLM cluster

Casualtek - Ransomchats dataset maintainers

⭐ Academic Project - University of Brescia, 2024-2025
