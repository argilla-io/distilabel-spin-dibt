pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu121
git clone https://github.com/uclaml/SPIN.git && cd SPIN
python -m pip install .
python -m pip install flash-attn==2.5.3 --no-build-isolation
huggingface-cli login --token $HF_API_TOKEN
pip install wandb
wandb login $WANDB_TOKEN