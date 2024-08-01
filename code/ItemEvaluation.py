# Using LLMs to evaluate the quality of CPS items
import time
import torch
import re
import json as js
import pandas as pd
import config

# OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseOutputParser
from langchain.prompts.chat import ChatPromptTemplate

# HF
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers import pipeline as hf_pipeline

# API key stored in key.py, and should NOT be committed
# TODO: add support for other api models called directly from this script
from key import OPENAI_KEY
from tqdm import tqdm
from random import randint
from argparse import ArgumentParser
from nltk import word_tokenize

from Prompts import item_eval_prompts


class CreativityScenarioItemEvaluationParser(BaseOutputParser):
    # the output should be formatted as json
    @staticmethod
    def parse(text: str) -> dict:
        #
        def fix_json(s: str):
            s = s[next(idx for idx, c in enumerate(s) if c in "{[") :]
            try:
                return js.loads(s)
            except js.JSONDecodeError as e:
                return js.loads(s[: e.pos])

        return js.dumps(fix_json(text))


def test_creative_problem_eval(prompt_idx: int, scenario: str, llm):
    # TODO: update with exemplars rated by humans using the new scale.
    prompts = item_eval_prompts

    output = llm.predict_messages(prompts[prompt_idx])

    output = CreativityScenarioItemEvaluationParser().parse(output.content)

    return output


def evaluate_scenarios(
    prompt_idx: int,
    output_file: str,
    model_name: str,
    llm,
    round: int,
    scenario_col: str,
    itemEvalOutputFile: str,
    itemGenOutputFile: str,
    itemEvalFrequencyPenalty: float,
    itemEvalPresencePenalty: float,
    itemEvalMaxTokens: int,
    itemEvalTemperature: float,
    itemEvalTopP: float,
):

    scenarios = pd.read_json(
        itemGenOutputFile,
    )
    scenarios[f"ratings_round_{round}"] = ""
    scenarios["item_eval_frequency_penalty"] = itemEvalFrequencyPenalty
    scenarios["item_eval_presence_penalty"] = itemEvalPresencePenalty
    scenarios["item_eval_max_tokens"] = itemEvalMaxTokens
    scenarios["item_eval_temperature"] = itemEvalTemperature
    scenarios["item_eval_top_p"] = itemEvalTopP
    scenarios["Evaluator"] = model_name
    for index, row in tqdm(scenarios.iterrows(), total=scenarios.shape[0]):
        evaluation = "None"
        time.sleep(10)
        for i in range(6):  # retry a maximum of x times
            try:
                evaluation = test_creative_problem_eval(
                    prompt_idx,
                    row[scenario_col],  # "creative_scenario_round_i"
                    llm,
                )
            except Exception as e:
                with open(config["logFile"], "w") as log:
                    print(e)
                    log.writelines(e)
                evaluation = "None"
                continue
            if evaluation != "None":
                break

        scenarios.at[index, f"ratings_round_{round}"] = evaluation

    # drop rows that failed quality control metrics
    scenarios = scenarios[scenarios[f"ratings_round_{round}"] != "None"]
    scenarios.to_json(
        itemEvalOutputFile,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str)  # the LLM used to evaluate items
    parser.add_argument("--task", type=str)
    parser.add_argument("--top_p", type=float)
    parser.add_argument("--frequency_penalty", type=float)
    parser.add_argument("--presence_penalty", type=float)
    parser.add_argument("--prompt_idx", type=int)
    parser.add_argument("--max_tokens", type=int)
    parser.add_argument("--output_file", type=str)  # the prefix to the output file name
    parser.add_argument("--round", type=int)
    parser.add_argument("--scenario_col", type=str)
    parser = parser.parse_args()
    try:
        task = parser.task
        model_name = parser.model_name
        temperature = 0  # output should be as close to deterministic as possible
        max_tokens = parser.max_tokens
        top_p = parser.top_p
        frequency_penalty = parser.frequency_penalty
        presence_penalty = parser.presence_penalty
        if model_name == "gpt-4" or model_name == "gpt-3.5-turbo":
            model_kwargs = {
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
            }
            llm = ChatOpenAI(
                model_name=model_name,
                openai_api_key=OPENAI_KEY,
                temperature=temperature,
                max_tokens=max_tokens,
                model_kwargs=model_kwargs,
            )

        else:
            model_kwargs = {
                "top_p": top_p,
                "temperature": temperature,
                "device_map": "auto",
                # "torch_dtype": torch.bfloat16, # don't use with 8 bit mode
            }
            tokenizer = AutoTokenizer.from_pretrained(model_name, **model_kwargs)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, load_in_4bit=True, **model_kwargs
            )
            pipeline = hf_pipeline(
                task="text-generation",
                model=model,
                tokenizer=tokenizer,
                batch_size=1,
                max_new_tokens=max_tokens,
                model_kwargs=model_kwargs,
            )
            llm = HuggingFacePipeline(pipeline=pipeline)

    except Exception:
        print("Model failed to initialize. Please check your API key.")
        exit(-1)
    if task == "evaluate_CPS":
        evaluate_scenarios(
            parser.prompt_idx,
            parser.output_file,
            parser.model_name,
            llm,
            parser.round,
            parser.scenario_col,
        )
    elif task == "evaluate_consequences":
        print("Consequences eval not implemented!")
        exit(-1)
    else:
        print("Not a valid task name!")
        exit(-1)
