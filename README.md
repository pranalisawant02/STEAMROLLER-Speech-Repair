# STEAMROLLER - Speech Repair Assistant


STEAMROLLER is a multi-agent speech repair system designed to improve the readability and fluency of speech affected by repetitions, prolongations, blocks, and filler words.

The system combines automatic speech recognition, NLP-based preprocessing, specialized repair agents, multi-agent evaluation, iterative feedback, and text-to-speech generation.

## Architecture

```text
Input Audio
    |
    v
Whisper ASR
    |
    v
NLP Cleaning
    |
    v
+----------------------------------+
|       Parallel Repair Agents     |
|                                  |
|  Meaning Preservation Agent      |
|  Grammar Optimized Agent         |
|  Concise Fluent Agent            |
|  Sequence Optimized Agent        |
+----------------------------------+
    |
    v
Candidate Outputs
    |
    v
Master Agent
    |
    +--> Meaning Preservation Score
    +--> Grammar Score
    +--> Fluency Score
    +--> Sequence Score
    |
    v
Best Candidate
    |
    +--> Threshold satisfied -> Final Output
    |
    +--> Threshold not satisfied
             |
             v
       Feedback Iteration
       (maximum 3 iterations)
    |
    v
Text-to-Speech
    |
    v
Repaired Audio
```

## Multi-Agent Design

STEAMROLLER uses four specialized rule-based repair agents:

* **Meaning Preservation Agent** - removes common filler words while preserving the original meaning.
* **Grammar Optimized Agent** - applies lightweight grammar and normalization corrections.
* **Concise Fluent Agent** - removes fillers, repeated words, and repeated characters to improve fluency.
* **Sequence Optimized Agent** - handles short restart fragments and repeated word sequences.

The agents are executed in parallel using Python's `ThreadPoolExecutor`.

Each agent generates an independent candidate repair. The candidates are then passed to the Master Agent for evaluation and selection.

## Master Agent

The Master Agent implements an iterative evaluation and feedback loop.

Candidate outputs are evaluated using four heuristic scores:

| Metric               | Weight |
| -------------------- | -----: |
| Meaning preservation |    50% |
| Grammar quality      |    30% |
| Concise fluency      |    15% |
| Sequence similarity  |     5% |

The combined score is compared against a threshold of **0.80**.

* If a candidate satisfies the threshold, the highest-scoring valid candidate is selected.
* If no candidate satisfies the threshold, the highest-scoring candidate is fed into another iteration.
* The process is bounded to a maximum of **3 iterations**.
* Intermediate candidate outputs and iteration scores are retained as execution state.

This provides an explicit candidate-generation, evaluation, selection, and feedback mechanism.

## Technology Stack

* Python
* Streamlit
* OpenAI Whisper
* NLTK
* PyTorch
* NumPy
* SciPy
* gTTS
* pyttsx3
* F5-TTS for optional voice cloning
* Flask and ngrok for the optional remote F5-TTS backend

## Project Structure

```text
STEAMROLLER/
|
|-- app.py
|-- audio_pipeline.py
|-- nlp_cleaner.py
|-- repair_agents.py
|-- repair_agents_runner.py
|-- master_agent.py
|-- tts_engine.py
|-- voice_clone.py
|-- kaggle_server_cell.py
|-- requirements.txt
|-- README.md
|-- .gitignore
```

### Main Components

**`app.py`**
Streamlit application and user interface.

**`audio_pipeline.py`**
Handles the audio processing and speech recognition pipeline.

**`nlp_cleaner.py`**
Performs transcript cleaning and preprocessing.

**`repair_agents.py`**
Contains the specialized speech repair agents.

**`repair_agents_runner.py`**
Runs the repair agents in parallel and collects their candidate outputs.

**`master_agent.py`**
Evaluates candidates, selects the best output, and manages the iterative feedback loop.

**`tts_engine.py`**
Handles text-to-speech generation.

**`voice_clone.py`**
Client for the optional remote F5-TTS voice cloning service.

**`kaggle_server_cell.py`**
Example Flask server setup for exposing an F5-TTS service from a GPU notebook.

## Installation

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Optional F5-TTS Voice Cloning

The default application can use local text-to-speech.

For voice cloning, STEAMROLLER can connect to an F5-TTS server running on a GPU notebook.

The local application reads the server address from the environment variable:

```text
KAGGLE_TTS_URL
```

### Windows PowerShell

```powershell
$env:KAGGLE_TTS_URL="https://YOUR-NGROK-URL"
streamlit run app.py
```

The Kaggle server setup is provided in:

```text
kaggle_server_cell.py
```

The GPU notebook/server must be running for the remote voice cloning feature to work.

## Evaluation and State

The current evaluation mechanism is an experimental heuristic harness rather than a production-scale benchmark.

The system retains:

* Original transcript
* Cleaned transcript
* Candidate outputs
* Individual quality scores
* Selected candidate
* Iteration history
* Final threshold status

This execution state allows the system to inspect intermediate decisions and repeat the repair process when the quality threshold is not met.

## Current Limitations

* Repair agents currently use deterministic NLP/rule-based techniques rather than LLM-based generation.
* Evaluation scores are heuristic and are not calibrated probabilities.
* The system is a research/demo prototype rather than a production deployment.
* Processing is primarily designed for uploaded or recorded audio rather than continuous real-time speech.
* F5-TTS voice cloning requires a separate GPU-backed service.

## Future Improvements

* LLM-based contextual repair agents
* Learned speech-repair evaluation metrics
* Larger benchmark datasets for quantitativ
