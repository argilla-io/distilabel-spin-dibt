# distilabel-spin-dibt

<div>
    <img src="https://cdn-uploads.huggingface.co/production/uploads/60420dccc15e823a685f2b03/aEzpD6gvn0xOrN2rNzpZI.webp">
</div>


<p align="center">
  <a href="https://github.com/argilla-io/distilabel">
    <img src="https://raw.githubusercontent.com/argilla-io/distilabel/main/docs/assets/distilabel-badge-light.png" alt="Built with Distilabel" width="200" height="32"/>
  </a>
</p>

SPIN experiments on the DIBT 10k ranked prompts.

This repository contains the instructions to run [SPIN](https://github.com/uclaml/SPIN) on a subset of the [DIBT/10k_prompts_ranked](https://huggingface.co/datasets/DIBT/10k_prompts_ranked) dataset: Those that have `avg_rating>=4` and `num_response>1`, making a total of 1832 records (which will then be splitted in 1648 for training and 184 for testing).

It contains the references to all the scripts to generate the datasets, the configuration files used for the training process and the setup used to run the model. The dataset generation was done using [distilabel==0.6.0](https://github.com/argilla-io/distilabel).

SPIN needs a specific format for the data to do the training, where the "real" data is the reference for the model to improve. As the dataset was made of prompts, we decided to generate these responses using [`mistral-large`](https://docs.mistral.ai/platform/endpoints/). The different iterations of the "generated" datasets were created using `distilabel` with `vllm`, using 2 A100 GPUs (just for speed, it should work with less computer power, just need to update the `--cuda-devices` and `--batch-size` arguments accordingly).

## Contribute to the DIBT prompt collective
This work shows the huge benefit of collecting high-quality prompts for LLM fine-tuning. If you want to support the OSS community with larger datasets, contribute to the [Prompt Collective initiative](https://huggingface.co/spaces/DIBT/prompt-collective-dashboard).

## Prepare the data

Initially, we create the reference dataset with the *real* responses being generated from `mistral-large`, using the following script:

- `generate_reference_spin.py`
    Script to generate the reference responses, uses `mistral-large`:

    Dataset: [argilla/10k_prompts_ranked_with_responses](https://huggingface.co/datasets/argilla/10k_prompts_ranked_with_responses)

### Experiment *top* prompts

The following are the steps to prepare the training data for SPIN, and the resulting datasets:

<details><summary> SPIN iter 0 </summary><hr>

- `generate_iter_spin.py`
    Script to generate the initial "generated" responses, from the SFT model that will then be fine-tuned.

    Dataset: [argilla/10k_prompts_ranked_sft_zephyr](https://huggingface.co/datasets/argilla/10k_prompts_ranked_sft_zephyr)

    Run the following:

    ```console
    python generate_iter_spin.py \
        --hf-apikey $HF_API_TOKEN \
        --source-dataset "DIBT/10k_prompts_ranked" \
        --new-dataset "argilla/10k_prompts_ranked_sft_zephyr" \
        --model-name "alignment-handbook/zephyr-7b-sft-full" \
        --batch-size 128 \
        --cuda-devices "0,1"
    ```

- `prepare_for_training.py`
    Generates the dataset that will be directly ingested in the `SPINTrainer`.

    Dataset: [argilla/10k_prompts_top_SPIN_iter0](https://huggingface.co/datasets/argilla/10k_prompts_top_SPIN_iter0)

    Running the following python script: 

    ```console
    python prepare_for_training.py \
        --portion top \
        --target-dataset argilla/10k_prompts_SPIN_iter0_zephyr_top
    ```

</details>


<details><summary> SPIN iter 1 </summary><hr>


- `generate_iter_spin.py`

    Regenerates the "generated" responses from the model in the previous iteration:

    ```console
    python generate_iter_spin.py \
        --hf-apikey $HF_API_TOKEN \
        --source-dataset "argilla/10k_prompts_SPIN_iter0_zephyr_top" \
        --new-dataset "argilla/10k_prompts_SPIN_iter1_zephyr_top_generated" \
        --model-name "plaguss/zephyr-7b-spin-iter0-v0" \
        --batch-size 128 \
        --cuda-devices "0,1"
    ```

    Dataset: [argilla/10k_prompts_top_SPIN_iter1_generated](https://huggingface.co/datasets/argilla/10k_prompts_top_SPIN_iter1_generated)

- `transform_iter_generated.py`

    The script transforms the generated responses to the format expected by SPIN trainer:

    ```console
    python transform_iter_generated.py \
        --real-dataset "argilla/10k_prompts_ranked_with_responses" \
        --generated-dataset "argilla/10k_prompts_SPIN_iter1_zephyr_top_generated" \
        --new-dataset "argilla/10k_prompts_SPIN_iter1_zephyr_top"
    ```

</details>


<details><summary> SPIN iter 2 </summary><hr>


- `generate_iter_spin.py`

    Regenerates the "generated" responses from the model in the previous iteration:

    ```console
    python generate_iter_spin.py \
        --hf-apikey $HF_API_TOKEN \
        --source-dataset "argilla/10k_prompts_SPIN_iter0_zephyr_top" \
        --new-dataset "argilla/10k_prompts_SPIN_iter2_zephyr_top_generated" \
        --model-name "plaguss/zephyr-7b-spin-iter1-v0" \
        --batch-size 128 \
        --cuda-devices "0,1"
    ```

    Dataset: [argilla/10k_prompts_top_SPIN_iter2_generated](https://huggingface.co/datasets/argilla/10k_prompts_top_SPIN_iter2_generated)

- `transform_iter_generated.py`

    The script transforms the generated responses to the format expected by SPIN trainer:

    ```console
    python transform_iter_generated.py \
        --real-dataset "argilla/10k_prompts_ranked_with_responses" \
        --generated-dataset "argilla/10k_prompts_SPIN_iter2_zephyr_top_generated" \
        --new-dataset "argilla/10k_prompts_SPIN_iter2_zephyr_top"
    ```

</details>

<details><summary> SPIN iter 3 </summary><hr>

- `generate_iter_spin.py`

    Regenerates the "generated" responses from the model in the previous iteration:

    ```console
    python generate_iter_spin.py \
        --hf-apikey $HF_API_TOKEN \
        --source-dataset "argilla/10k_prompts_SPIN_iter0_zephyr_top" \
        --new-dataset "argilla/10k_prompts_SPIN_iter3_zephyr_top_generated" \
        --model-name "plaguss/zephyr-7b-spin-iter2-v0" \
        --batch-size 128 \
        --cuda-devices "0,1"
    ```

    Dataset: [argilla/10k_prompts_top_SPIN_iter3_generated](https://huggingface.co/datasets/argilla/10k_prompts_top_SPIN_iter3_generated)

- `transform_iter_generated.py`

    The script transforms the generated responses to the format expected by SPIN trainer:

    ```console
    python transform_iter_generated.py \
        --real-dataset "argilla/10k_prompts_ranked_with_responses" \
        --generated-dataset "argilla/10k_prompts_SPIN_iter3_zephyr_top_generated" \
        --new-dataset "argilla/10k_prompts_SPIN_iter3_zephyr_top"
    ```

</details>


## Fine tune using SPIN

The following steps are almost a copy from the [SPIN](https://github.com/uclaml/SPIN) repository, take a look there for more information.

### Runpod

We used Runpod with the following setup:

- 4 A100 80Gb.
- 500Gb container and volume.
- Base image with CUDA 12.1.

### Once with the POD running

These are the steps outlined in the SPIN repo, you can run them by running the script in `scripts/setup.sh`:

```console
pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu121
```

Clone and install the repo from source:

```console
git clone https://github.com/uclaml/SPIN.git && cd SPIN
```

Install package and flash-attn

```console
python -m pip install .
python -m pip install flash-attn==2.5.3 --no-build-isolation
```

Log to huggingface:

```console
huggingface-cli login --token $HF_API_TOKEN
```

Log to wandb:

```console
pip install wandb
wandb login $WANDB_TOKEN
```

And update the WANDB variables to keep track of the experiments:

```console
export WANDB_ENTITY="argilla-io"
export WANDB_PROJECT="dibt-spin-zephyr"
export WANDB_NAME="zephyr-7b-spin-iter0-v0"
```

After the previous step, replace the config file of the model to run, and the `finetune.sh` script, and start the training process:

```console
bash scripts/finetune.sh
```

### Weights and Biases runs

<details><summary> DIBT 10k *Top* subset </summary><hr>

- [argilla-io/dibt-top-spin-iter0-zephyr](https://wandb.ai/argilla-io/dibt-spin-zephyr/runs/439olh1m?nw=nwuserplagussargilla)

- [argilla-io/dibt-top-spin-iter1-zephyr](https://wandb.ai/argilla-io/dibt-spin-zephyr/runs/q938reyu?nw=nwuserplagussargilla)

- [argilla-io/dibt-top-spin-iter2-zephyr](https://wandb.ai/argilla-io/dibt-spin-zephyr/runs/q40amnp0?nw=nwuserplagussargilla)

- [argilla-io/dibt-top-spin-iter3-zephyr](https://wandb.ai/argilla-io/dibt-spin-zephyr/runs/u8znanpw?nw=nwuserplagussargilla)

</details>
