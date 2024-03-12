"""
$ python dibt/synthetic_evaluator.py
"""

import re
from typing import List

from textwrap import dedent
from typing import TypedDict, Optional
from dataclasses import dataclass
from distilabel.tasks import TextGenerationTask, Prompt


class Rating(TypedDict):
    """A `TypedDict` representing a rating."""

    value: int
    description: str


class PromptEvaluatorOutput(TypedDict):
    """A `TypedDict` representing the output of an `PromptEvaluationTask`."""

    rating: float
    rationale: str


prompt_evaluator = """{task_description}
{ratings}

This is the prompt:
{input}

Your answer must be in the following format:

<rating>[1-5]</rating>
<rationale>your rationale</rationale>

Please rate the prompt and provide a rationale for your rating."""


@dataclass
class PromptEvaluationTask(TextGenerationTask):
    """Rough translation from the guidelines for the labelling task:
    https://dibt-prompt-collective.hf.space/dataset/f31dabc5-12d5-4845-8361-d41be905d808/settings
    to a distilabel task.
    """

    ratings: List[Rating] = None
    task_description: str = None
    system_prompt: str = (
        "You are an AI prompt evaluator focused on rating prompts that are clear, interesting and complex for fine-tuning open source LLMs."
    )

    def generate_prompt(self, input: str) -> "Prompt":
        render_kwargs = {
            "task_description": self.task_description,
            "ratings": self.ratings,
            "input": input,
        }
        return Prompt(
            system_prompt=self.system_prompt,
            formatted_prompt=prompt_evaluator.format(**render_kwargs),
        )

    @classmethod
    def for_overall_quality(
        cls,
        system_prompt: Optional[str] = None,
        task_description: Optional[str] = None,
        ratings: Optional[List[Rating]] = None,
    ) -> "UltraFeedbackTask":
        kwargs = {}
        if system_prompt is not None:
            kwargs.update({"system_prompt": system_prompt})
        if task_description is None:
            task_description = """You need to assign a rating to each prompt thinking about the complexity for an assistant and if the intent is clear. A very good prompt is one that is challenging but also very clear in the intent of the user.

An example of a good prompt involves the following aspects:
- The intent of the user is clear.
- The question, instruction or task for the assistant is challenging or interesting because it involves solving a complex problem, reasoning, involving being creative, etc.

In the case that you feel unequipped of rating a specific prompt, please rate it with -1.

**Scoring**: Rate outputs 1 to 5 based on the following aspects:
"""
        kwargs.update({"task_description": task_description})

        if ratings is None:
            ratings = [
                Rating(
                    value=1,
                    description="**Very Bad**:\n The prompt doesn't communicate its purpose, is non-sensical or is in a language other than English. The prompt assumes the usage of tools or capabilities that don’t apply to this model, like generating an image or scraping a website.",
                ),
                Rating(
                    value=2,
                    description="**Bad**:\n Suggests a goal but lacks clarity and coherence.",
                ),
                Rating(
                    value=3,
                    description="**Ok**:\n The intent is understandable, but it's missing information to complete the task.",
                ),
                Rating(
                    value=4,
                    description="**Good**:\n Presents a clear goal and necessary information, effectively directing the AI, but the prompt could be more specific.",
                ),
                Rating(
                    value=5,
                    description="**Very Good**:\n Comprehensive and explicit, leaving no room for ambiguity. Perfectly guides the AI and includes details.",
                ),
            ]
            written_ratings = "\n".join(
                [f"{rating['value']}. {rating['description']}" for rating in ratings]
            )
        kwargs.update({"ratings": written_ratings})
        return cls(**kwargs)

    def parse_output(self, output: str) -> "CritiqueTaskOutput":  # type: ignore
        """Parses the output of the model into the desired format."""
        pattern = r"<rating>(.*?)</rating>\s*<rationale>(.*?)</rationale>"
        match = re.findall(pattern, output, re.DOTALL)
        if match:
            return PromptEvaluatorOutput(
                rating=float(match[0][0]),
                rationale=match[0][1].strip(),
            )


if __name__ == "__main__":
    import os

    from distilabel.pipeline import Pipeline
    from distilabel.llm import OpenAILLM
    from datasets import load_dataset, concatenate_datasets
    from distilabel.dataset import DatasetCheckpoint

    from dotenv import load_dotenv

    load_dotenv()

    OPENAI_API_TOKEN = os.getenv("OPENAI_API_TOKEN")
    HF_API_TOKEN = os.getenv("HF_API_TOKEN")
    # DATASET_CONFIGURAITON = "DIBT/10k_prompts_ranked"
    MAX_ROWS = 10_000
    MIN_NUM_RESPONSES = 1
    NEW_DATASET_NAME = "argilla/DIBT_prompts_ranked_synthetic_n3_s1000"

    latest_dataset = load_dataset(NEW_DATASET_NAME, split="train")
    dataset = load_dataset("DIBT/10k_prompts_ranked", split="train").rename_column(
        "prompt", "input"
    )
    dataset = dataset.filter(
        lambda x: int(x["num_responses"]) >= MIN_NUM_RESPONSES, keep_in_memory=True
    )
    dataset = dataset.select(range(MAX_ROWS))
    checkpoint_strategy = DatasetCheckpoint(
        strategy="hf-hub",
        extra_kwargs={
            "repo_id": NEW_DATASET_NAME,
            "token": HF_API_TOKEN,
            "private": True,
            "split": "train",
        },
        save_frequency=100,
    )

    pipe = Pipeline(
        generator=OpenAILLM(
            model="gpt-4-1106-preview",  # gpt-4 turbo
            task=PromptEvaluationTask.for_overall_quality(),
            max_new_tokens=512,
            num_threads=8,
            api_key=OPENAI_API_TOKEN,
            temperature=0.3,
        )
    )
    new_ds = pipe.generate(
        dataset,
        num_generations=1,
        batch_size=16,
        checkpoint_strategy=checkpoint_strategy,
    )

    updated_dataset = load_dataset(NEW_DATASET_NAME, split="train")

    updated_dataset = concatenate_datasets([updated_dataset, latest_dataset])

    updated_dataset.push_to_hub(
        NEW_DATASET_NAME, use_auth_token=HF_API_TOKEN, split="train"
    )
