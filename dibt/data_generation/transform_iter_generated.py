"""
Script to generate the initial SPIN iter0 dataset for DIBT 10K ranked.

"""

from datasets import load_dataset, concatenate_datasets, Dataset

def prepare_for_spin(example):
    return {
        "real": [
            {"role": "user", "content": example["input"]}, 
            {"role": "assistant", "content": example["real"][0]}
        ],
        "generated": [
            {"role": "user", "content": example["input"]}, 
            {"role": "assistant", "content": example["generated"][0]}
        ]
    }


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Prepare the dataset for training using SPIN. Start from the 10k ranked dataset and add the synthetic responses as real.")
    parser.add_argument("--real-dataset", type=str, default="argilla/10k_prompts_ranked_with_responses")
    parser.add_argument("--generated-dataset", type=str, default="argilla/10k_prompts_top_SPIN_iter1_generated")
    parser.add_argument("--new-dataset", type=str, default="argilla/10k_prompts_SPIN_iter1")
    # 1) Create a new variable to allow generating the dataset up to a point.

    args = parser.parse_args()

    HF_API_TOKEN = os.getenv("HF_API_TOKEN")
    if not HF_API_TOKEN:
        raise ValueError("You need to set the HF_API_TOKEN environment variable to push the dataset to the hub.")

    ds_real = load_dataset(args.real_dataset, split="train")
    ds_generated = load_dataset(args.generated_dataset, split="train")

    columns = ["input", "generations"]
    df_real = ds_real.to_pandas()
    df_generated = ds_generated.to_pandas()

    ds_for_spin = Dataset.from_pandas(
        df_generated[columns].merge(
            df_real[columns], on="input"
        ).rename(columns={"generations_x": "generated", "generations_y": "real"}),
        preserve_index=False
    )

    print(args)
    ds_for_spin = ds_for_spin.map(prepare_for_spin, remove_columns=["input"])
    ds_for_spin = ds_for_spin.train_test_split(test_size=0.1, seed=42)
    ds_for_spin.push_to_hub(args.new_dataset, token=HF_API_TOKEN, private=True)
