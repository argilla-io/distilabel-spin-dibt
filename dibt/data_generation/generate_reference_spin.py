"""Script to generate responses for DIBT 10K ranked.

pip install distilabel
pip install mistralai==0.0.11
"""
import os

from datasets import Dataset, load_dataset
from distilabel.pipeline import Pipeline
from distilabel.tasks import TextGenerationTask
from distilabel.llm.mistralai import MistralAILLM

from huggingface_hub import login

from distilabel.tasks import TextGenerationTask
from distilabel.tasks.prompt import Prompt
from distilabel.dataset import DatasetCheckpoint

from dataclasses import dataclass


def get_dataset() -> Dataset:
    return load_dataset("DIBT/10k_prompts_ranked", split="train")


@dataclass
class SPINTextGenerationTask(TextGenerationTask):
    """Generic task to generate the prompts following SPIN.
    [SPIN](https://github.com/uclaml/SPIN/blob/main/spin/generate.py)
    """
    system_prompt: str = ""

    def generate_prompt(self, input: str) -> Prompt:
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=input
        )



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mistralai-apikey", type=str, default=None)
    parser.add_argument("--hf-apikey", type=str, default=None, help="Your HuggingFace API key with **WRITE** permission, otherwise it cannot push to hub")
    # 1) Create a new variable to allow generating the dataset up to a point.

    args = parser.parse_args()

    HF_API_TOKEN = args.hf_apikey or os.getenv("HF_API_TOKEN")
    MISTRALAI_API_KEY = args.mistralai_apikey or os.getenv("MISTRALAI_API_KEY")
    SAVE_FREQ = 500

    # if args.push_to_hub:
    # Log to huggingface hub
    login(token=HF_API_TOKEN)

    dataset = get_dataset()
    dataset = dataset.rename_column("prompt", "input")

    num_generations = len(dataset)
    print("num_generations", num_generations)

    DATASET_NAME = "argilla/10k_prompts_ranked_with_responses"

    checkpoint = DatasetCheckpoint(
        strategy="hf-hub",
        extra_kwargs={
            "repo_id": DATASET_NAME,
            "token": HF_API_TOKEN,
            "private": True,
            "split": "train"
        },
        save_frequency=SAVE_FREQ
    )

    print(f"Save frequency: every {SAVE_FREQ} rows.")

    # endpoints: https://docs.mistral.ai/platform/endpoints/
    pipe_generation = Pipeline(
        generator=MistralAILLM(
            model="mistral-large-2402",
            task=SPINTextGenerationTask(),
            api_key=MISTRALAI_API_KEY,
            max_tokens=2048,
            num_threads=8,
            temperature=1
        )
    )
    dibt_10k_ranked_responses = pipe_generation.generate(
        dataset=dataset,
        num_generations=1,
        batch_size=16,
        checkpoint_strategy=checkpoint,
    )
