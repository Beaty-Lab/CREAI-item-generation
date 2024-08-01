# The creative psychometric item generator: a framework for item generation and validation using large language models

## Abstract
Increasingly, large language models (LLMs) are being used to automate workplace processes requiring a high degree of creativity. While much prior work has examined the creativity of LLMs, there has been little research on whether they can generate valid creativity assessments for humans despite the increasingly central role of creativity in modern economies. We develop a psychometrically inspired framework for creating test items (questions) for a classic free-response creativity test: the creative problem solving (CPS) task. Our framework, the creative psychometric item generator (CPIG), uses a mixture of LLM-based item generators and evaluators to iteratively develop new prompts for writing CPS items, such that items from later iterations will elicit more creative responses from test takers. We find strong empirical evidence that CPIG generates valid and reliable items and that this effect is not attributable to known biases in the evaluation process. Our findings have implications for employing LLMs to automatically generate valid and reliable creativity tests for humans and AI.

## Repository Description
Contains code and data for reproducing our results from our CREAI paper.
## Setup
### Requirements
1. Python 3.x, we reccomend the latest version from Anaconda.
2. pandas
3. numpy
4. transformers
5. langchain
6. openai
7. anthropic
8. accelerate
9. bitsandbytes
10. pytorch
11. nltk
12. readability
13. tqdm
14. peft
15. evaluate
16. scipy

You can install these into your anaconda enviroment by simply using:

```pip install requirements.txt```
### Important scripts
1. `RunExperiment.py`, the main driver script for running trials, you should run this script to reproduce results.
2. `ItemGeneration.py`, code for generating CPS items.
3. `GenerateCPSResponses.py`, code for generating CPS item responses.
4. `RLPS_RoBERTa.py`, code for finetuning the originality model. Note that this is slightly modified from the original to include `peft`.
5. `SelectItemGenShots.py`, code for shot selection methods.
6. `ItemEvaluation.py`, an experimental script for generating LLM evaluations of LLM items, not used in the final submission.
7. `Prompts.py`, all prompts used, stored as lists that may be added onto.
8. `config.py`, defines the paramters used for an experiment, discussed in detail below.
### Setting up the config
1. `random_seed`: the seed for numpy / pytorch / huggingface. This is ingnored for Claude.
2. `numIter`: how many iterations to run.
3. `itemGenModelName`: the name of the item generator. Use either a huggingface model id or `claude-3` for Claude-haiku.
4. `itemResponseGenModelName`: the name of the item response generator. Use either a huggingface model id or `claude-3` for Claude-haiku.
5. `itemGenPromptIdx`: the index of the prompt for item generation in `Prompts.py`.
6. `itemResponseGenPromptIdx`: the index of the prompt for item response generation in `Prompts.py`.
7. `itemGenMaxTokens`: how many tokens to cap item generation.
8. `itemResponseGenMaxTokens`: how many tokens to cap item response generation.
9. `demographicsFile`: The file to fetch demographic / psychometric data for demographic / psychometric prompts. We store the csvs we used under `./creativity-item-generation/optimize_item_gen_prompt/data/`. Use `PsychometricData.csv` for psychometric prompts, and `DemographicData.csv` for demographic prompts. Set to `None` for the no context prompt.
10. `itemGenOutputFile`, `itemResponseGenOutputFile`: where to save generated items and item response, stored in separate json files. Ideally these should point to the same folder, but this is not a requriement.
11. `numItemGenerationAttempts`: how many times to retry generation if it fails a quality control check.
12. `itemResponseOriginalityModelDir`: the directory of the pre-trained originality scoring model. We are unable to provide the pre-trained weights, but the authors from the cited work can be contacted to obtain them.
13. `itemGenNumShots`: The size of `k` to use as exemplars.
14. `shotSelectionAlgorithm`: The algorithm used for shot selection, one of `random`, `greedy`, `constraint satisfaction`.
15. `numResponsesPerItem`: how many LLM responses to generate per item.

## Running the code
1. Install the listed packages into a Python enviroment.
2. Obtain a Claude API key, the driver expects a `key.py` in the same directory with the key stored as a string named `ANTHROPIC_KEY`. You may additionally obtain an OpenAI key and store it under `OPENAI_KEY`.
3. Set the hyperparamter settings under `config.py`, and create the directory where results will be stored.
4. Run `RunExperiment.py`

Please refer to the authors of the originality scorer (referenced in the paper) to access the weights of the scoring model. Note that you must have the weights saved locally for originality scoring.

## Citation
If you find this work helpful, please cite us:
```
@inproceedings{laverghetta_luchini_linell_reiter-palmon_beaty_2024,  
    title={The creative psychometric item generator: a framework for item generation and validation using large language models},  
    booktitle={CREAI 2024: International Workshop on Artificial Intelligence and Creativity.},  publisher={CEUR-WS},  
    author={Laverghetta, Antonio and Luchini, Simone and Linell, Averie and Reiter-Palmon, Roni and Beaty, Roger},  
    year={2024}
}
```
