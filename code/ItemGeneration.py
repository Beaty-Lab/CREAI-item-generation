import time
import numpy as np
import re
import pandas as pd
from config import config

# OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import _convert_to_message

# HF
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers import pipeline as hf_pipeline


from langchain.schema import BaseOutputParser, StrOutputParser, OutputParserException
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain.output_parsers import RetryOutputParser
from langchain.prompts.chat import ChatPromptTemplate

# API key stored in key.py, and should NOT be committed
# TODO: add api support for other API models run directly from this script
from key import OPENAI_KEY
from tqdm import tqdm
from readability import Readability
from argparse import ArgumentParser
from nltk import word_tokenize

from Prompts import item_gen_prompts, wordlist_gen_prompts


# class for storing and manipulating prompts
class PromptGenerator:
    @staticmethod
    def make_consequence_prompt():
        consequences_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a scenario writer."),
                (
                    "human",
                    """Think of a scenario the would change the way human life or the world works. The scenario should alter people's daily lives in important ways. Please describe only the scenario, and don't hint at any potential implications of the scenario. Please describe this scenario in 12 words at most.

        ###

        Scenario:""",
                ),
            ]
        )
        return consequences_prompt

    # NOTE: few shots are currently baked into prompt
    @staticmethod
    def make_creative_scenario_wordlist_generation_prompt(wordlist_prompt_idx: int):
        prompts = wordlist_gen_prompts
        creative_scenario_wordlist_generation_prompt = ChatPromptTemplate.from_messages(
            prompts[wordlist_prompt_idx]
        )
        return creative_scenario_wordlist_generation_prompt

    """
    Generates a prompt for creative scenario generation based on a given word list.

    Parameters:
        word_list (str): A list of 4 words, consisting of 2 names, a place, and an action.

    Returns:
        str: The generated prompt for creative scenario generation.
    """

    @staticmethod
    def make_creative_scenario_generation_prompt(scendario_prompt_idx: int):
        scenario_base_prompts = item_gen_prompts
        creative_scenario_generation_prompt = ChatPromptTemplate.from_messages(
            scenario_base_prompts[scendario_prompt_idx],
        )
        return creative_scenario_generation_prompt


class ConsequencesItemParser(BaseOutputParser):
    def parse(self, text: str) -> dict:
        js_output = {
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "output": text,
            "item_type": "consequences",
        }
        return js_output


class CreativeWordlistItemParser(BaseOutputParser):
    def parse(self, text: str) -> dict:
        text = text.split("\n\n")
        text = [re.sub(r"([0-9.])+", "", t) for t in text]
        text = [t.strip("\n").strip(" ") for t in text]
        js_output = {
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "output": text,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "item_type": "creative_scenario",
        }
        return js_output


class CreativityScenarioItemParser(BaseOutputParser):
    # TODO: is it possible for the user to specify some of the output formatitng, like the forbidden strings?
    # Add this to config file
    # TODO: can we pass the parsing exception into the retry parser, add it to the retry prompt?
    @staticmethod
    def parse(text: str) -> str:

        forbidden_strings = [
            "On the one hand",
            "On the other hand",
            "dilemma",
            "must navigate",
            "must decide",
            "has to decide",
            "is torn between",
        ]

        # Remove intervening newlines
        text = re.sub("\n", "", text)
        text = re.sub("\t", "", text)

        if text is None:
            print("Empty string generated.")
            raise OutputParserException("Empty string generated.")

        # remove all text after stop sequence
        if "I am finished with this scenario." not in text:
            print("Termination string not found.")
            raise OutputParserException("Termination string not found.")
        else:
            head, sep, tail = text.partition("I am finished with this scenario.")
            text = head

        # remove phrases indicating LLM is "spelling out" solution
        for f in forbidden_strings:
            if f in text:
                print("Scenario contains forbidden string.")
                raise OutputParserException("Scenario contains forbidden string.")

        readability = Readability(text)
        if len(word_tokenize(text)) < 140:  # drop scenarios that are too short
            print("Scenario too short.")
            raise OutputParserException("Scenario too short.")

        elif (
            readability.flesch().score < 45
        ):  # based on some initial feedback on the results
            print("Scenario too difficult to read.")
            raise OutputParserException("Scenario too difficult to read.")

        text = text.strip("\n").strip(" ")
        return text  # , "OK"

    @staticmethod
    def get_format_instructions() -> str:
        return "Respond in no more than 1 paragraph."


# test the chain
def test_consequences():
    prompt = PromptGenerator.make_consequence_prompt()
    chain = prompt | llm | ConsequencesItemParser()
    result = chain.invoke(
        {"": ""}
    )  # invoking a chain requires passing an input for the template, which can be a blank dict
    return result


def test_creative_wordlist_generation(prompt_idx: int, llm):
    prompt = PromptGenerator.make_creative_scenario_wordlist_generation_prompt(
        prompt_idx
    )  # the prompt type
    chain = prompt | llm | CreativeWordlistItemParser()
    word_list = chain.invoke({"": ""})
    return word_list


def test_creative_problem(
    word_list,
    prompt_idx: int,
    llm,
    scenario_names: str,
    previous_llm_output=None,
    topic_from_file=None,
    ratings_from_file=None,
    item_shots: list = None,
    numAttempts: int = 1,
):
    parser = CreativityScenarioItemParser()
    retry_parser = RetryOutputParser.from_llm(
        parser=parser, llm=llm, max_retries=numAttempts
    )
    # when true, will use AI feedback to improve the model outputs
    if previous_llm_output is not None:
        prompt = PromptGenerator.make_creative_scenario_generation_prompt(
            prompt_idx
        )  # the prompt type
        human_query = prompt.messages[1].prompt.template
        prompt.messages.pop()

        # I think put them as one message where the human lists some example items
        # TODO: make sure the items are properly formatted
        item_shots = _convert_to_message(
            (
                "human",
                "\nHere are some more examples of high quality scenarios from other authors. Use these scenarios as guidance, but avoid drawing from them too heavily when developing your own:\n"
                + "\n###\n".join(item_shots)
                + f"""\n###\nWord list:
                    {word_list}

                    ###

                    Scenario:""",
            )
        )
        prompt.messages.insert(1, item_shots)

        # add AI feedback, if it exists
        if ratings_from_file is not None:
            # add previous output to the prompt
            prompt.messages.insert(2, _convert_to_message(("ai", "{ai_output}")))
            # add the LLM evaluation to the prompt
            prompt.messages.insert(
                3,
                _convert_to_message(
                    (
                        "human",
                        """
                    Here is some feedback for your scenario:
                    {ai_feedback}

                    Please revise your scenario, and try improve your score in each category. Remember, maximizing the scores doesn't mean your scenario is better. Also, please don't make all your edits at the end of the scenario, spread them throughout.
                    
                    ###
                    
                    """
                        + human_query,
                    )
                ),
            )
        completion_chain = (
            prompt | llm | StrOutputParser()
        )  # StrOutputParser grabs the content field from chat models
        validation_chain = RunnableParallel(
            completion=completion_chain, prompt_value=prompt
        ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x))

        if ratings_from_file is not None:
            final_prompt = prompt.format(
                **{
                    "word_list": word_list,
                    "ai_output": previous_llm_output,
                    "ai_feedback": ratings_from_file,
                }
            )
            result = validation_chain.invoke(
                {
                    "word_list": word_list,
                    "ai_output": previous_llm_output,
                    "ai_feedback": ratings_from_file,
                }
            )
            print(result)
        else:
            final_prompt = prompt.format(
                **{
                    "word_list": word_list,
                }
            )
            result = validation_chain.invoke(
                {
                    "word_list": word_list,
                }
            )
            print(result)

        return result, final_prompt
    else:
        prompt = PromptGenerator.make_creative_scenario_generation_prompt(
            prompt_idx
        )  # the prompt type

        # StrOutputParser grabs the content field from chat models
        completion_chain = prompt | llm | StrOutputParser()

        # We try to regenerate a few times if the LLM fails validation
        # Should we be unable to fix the scenario, we return "None", these get dropped later
        validation_chain = RunnableParallel(
            completion=completion_chain, prompt_value=prompt
        ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x))

        final_prompt = prompt.format(word_list=word_list)
        result = validation_chain.invoke({"word_list": word_list})
        return result, final_prompt


# cookbooks for item gen
def create_wordlists(prompt_idx: int, output_file: str, llm):
    js_array = []
    for i in tqdm(range(5)):  # each wordlist call creates 10 lists
        time.sleep(2)  # rate limit
        result = test_creative_wordlist_generation(prompt_idx, llm)
        js_array.append(result)

    df = (
        pd.json_normalize(js_array)
        .explode("output")
        .reset_index()
        .drop("index", axis=1)
    )
    df.to_csv(f"{output_file}", sep="\t")


def create_scenarios(
    prompt_idx: int,
    model_name: str,
    llm,
    round,
    max_tokens: int,
    presence_penalty: float,
    frequency_penalty: float,
    temperature: float,
    top_p: float,
    itemGenOutputFile: str,
    numItemGenerationAttempts: int,
    input_file: str = None,
    wordlist_file: str = None,
    item_shots: list = None,
):
    # when true, will use add new items to an existing file
    if input_file != None and round >= 1:
        assert item_shots != None

        input_file = pd.read_json(input_file)
        if f"ratings_round_{round-1}" not in input_file.columns:
            input_file[f"ratings_round_{round-1}"] = None

        input_file[f"creative_scenario_round_{round}"] = ""
        input_file[f"prompt_round_{round}"] = ""
        for index, row in tqdm(input_file.iterrows(), total=input_file.shape[0]):
            result = "None"
            if (
                model_name == "gpt-4"
                or model_name == "gpt-3.5-turbo"
                or model_name == "google"
                or model_name == "claude-3"
            ):
                time.sleep(3)
            scenario_names = row["word_list"].split(",")[::2]
            scenario_names = [
                re.sub(r"([0-9]{1}\.)", "", s).strip() for s in scenario_names
            ]
            # generation may fail due to google API filters
            try:
                result, prompt = test_creative_problem(
                    row["word_list"],
                    prompt_idx,
                    llm,
                    scenario_names,
                    row[f"creative_scenario_round_{round-1}"],
                    ratings_from_file=row[f"ratings_round_{round-1}"],
                    item_shots=item_shots,
                    numAttempts=numItemGenerationAttempts,  # keep on generating scenarios until the model passes all quality control checks
                )
                print(result)

            except Exception as e:
                with open(config["logFile"], "a") as log:
                    print(e)
                    log.writelines(str(e) + "\n")
                result = np.nan
                prompt = np.nan
                continue

            input_file.at[index, f"creative_scenario_round_{round}"] = result
            input_file.at[index, f"prompt_round_{round}"] = prompt

        # drop rows that failed quality controls
        # TODO: over multiple rounds, there could be many scenarios for the same wordlist + topic
        # if at any point one of those comes back null, so long as prior iterations weren't null
        # the row SHOULD NOT be deleted!
        # change the drop to account for this, will need some careful engineering
        input_file = input_file[
            input_file[f"creative_scenario_round_{round}"] != "None"
        ]
        input_file = input_file[input_file[f"creative_scenario_round_{round}"] != ""]
        input_file = input_file[input_file[f"creative_scenario_round_{round}"] != None]
        input_file.dropna(subset=f"creative_scenario_round_{round}", inplace=True)
        input_file.to_json(itemGenOutputFile, orient="records")
        with open(config["logFile"], "a") as log:
            print(f"Item gen finished, total items: {len(input_file)}")
            log.writelines(f"Item gen finished, total items: {len(input_file)}\n")
    elif input_file == None and round == 0:
        # path for a fresh round of item generation without evalution
        wordlists = pd.read_csv(wordlist_file, sep="\t")
        wordlists.rename({"output": "word_list"}, axis=1, inplace=True)
        wordlists[f"creative_scenario_round_{round}"] = ""
        wordlists[f"prompt_round_{round}"] = ""
        wordlists_with_s = pd.DataFrame()
        for index, row in tqdm(wordlists.iterrows(), total=wordlists.shape[0]):
            result = "None"  # keep on generating scenarios until the model passes all quality control checks
            if (
                model_name == "gpt-4"
                or model_name == "gpt-3.5-turbo"
                or model_name == "google"
                or model_name == "claude-3"
            ):
                time.sleep(5)
            # grab just the names in the wordlist, need for preprocessing
            scenario_names = row["word_list"].split(",")[::2]
            scenario_names = [
                re.sub(r"([0-9]{1}\.)", "", s).strip() for s in scenario_names
            ]
            # generation may fail due to google API filters
            try:
                result, prompt = test_creative_problem(
                    row["word_list"],
                    prompt_idx,
                    llm,
                    scenario_names,
                    numAttempts=numItemGenerationAttempts,  # keep on generating scenarios until the model passes all quality control checks
                )
                print(result)
            except Exception as e:
                with open(config["logFile"], "a") as log:
                    print(e)
                    log.writelines(str(e) + "\n")
                result = np.nan
                prompt = np.nan
                continue

            new_scenario = pd.DataFrame(
                {
                    f"creative_scenario_round_{round}": result,
                    f"prompt_round_{round}": prompt,
                    "item_gen_model_name": model_name,
                    "item_gen_max_tokens": max_tokens,
                    "item_gen_presence_penalty": presence_penalty,
                    "item_gen_frequency_penalty": frequency_penalty,
                    "item_gen_temperature": temperature,
                    "item_type": row["item_type"],
                    "item_gen_top_p": top_p,
                    "word_list": row["word_list"],
                },
                index=[0],
            )
            wordlists_with_s = pd.concat((wordlists_with_s, new_scenario))

        with open(config["logFile"], "a") as log:
            print(f"Item gen finished, total items {len(wordlists_with_s)}")
            log.writelines(f"Item gen finished, total items {len(wordlists_with_s)}\n")
        # drop rows that failed quality control metrics
        wordlists_with_s.reset_index(drop=True, inplace=True)
        wordlists_with_s = wordlists_with_s[
            wordlists_with_s[f"creative_scenario_round_{round}"] != "None"
        ]
        wordlists_with_s = wordlists_with_s[
            wordlists_with_s[f"creative_scenario_round_{round}"] != ""
        ]
        wordlists_with_s = wordlists_with_s[
            wordlists_with_s[f"creative_scenario_round_{round}"] != None
        ]
        wordlists_with_s.dropna(subset=f"creative_scenario_round_{round}", inplace=True)
        wordlists_with_s.to_json(
            itemGenOutputFile,
            orient="records",
        )
    else:
        print("Unsupported combination of arguments!")


# test prompt X number of times, and save in df
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top_p", type=float)
    parser.add_argument("--frequency_penalty", type=float)
    parser.add_argument("--presence_penalty", type=float)
    parser.add_argument("--prompt_idx", type=int)
    parser.add_argument("--max_tokens", type=int)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--wordlist_file", type=str, default=None)
    parser = parser.parse_args()
    try:
        task = parser.task
        model_name = parser.model_name
        temperature = parser.temperature
        max_tokens = parser.max_tokens
        top_p = parser.top_p
        frequency_penalty = parser.frequency_penalty
        presence_penalty = parser.presence_penalty
        batch_size = parser.batch_size
        input_file = parser.input_file
        wordlist_file = parser.wordlist_file
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
                model_name, load_in_8bit=True, **model_kwargs
            )
            pipeline = hf_pipeline(
                task="text-generation",
                model=model,
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_new_tokens=max_tokens,
                model_kwargs=model_kwargs,
            )
            llm = HuggingFacePipeline(pipeline=pipeline)

    except Exception:
        print("Model failed to initialize. Please check your API key.")
        exit(-1)
    # scenario generation should be done using the driver script, not from here
    # if task == "scenario_generation":
    #     create_scenarios(
    #         parser.prompt_idx,
    #         parser.output_file,
    #         parser.model_name,
    #         llm,
    #         input_file,
    #         wordlist_file,
    #         presence_penalty=presence_penalty,
    #     )
    if task == "wordlist generation":
        create_wordlists(parser.prompt_idx, parser.output_file, llm)
    else:
        print("Not a valid task name!")
        exit(-1)
