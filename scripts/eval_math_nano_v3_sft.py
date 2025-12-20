from nemo_skills.pipeline.cli import generate, wrap_arguments, eval

cluster = "slurm"
output_dir = "/lustre/fsw/portfolios/llmservice/users/yachen/AceMath/Skills/nano-v3-sft-eval"
gpus=8
server_nodes=2
i=0
model_name = "nano-v3-sft"
model_path = "/lustre/fsw/portfolios/llmservice/projects/llmservice_fm_text/users/yachen/AceMath/checkpoint/nvidia-nemotron-3-nano-30b-a3b-sft"
eval(
    ctx=wrap_arguments(
        "++inference.tokens_to_generate=120000 "
        "++inference.temperature=1.0 "
        "++inference.top_p=1.0 "
    ),
    cluster=cluster,
    expname=f"{model_name}-no-python",
    model=model_path,
    server_type='vllm',
    server_gpus=8,
    num_chunks=4,
    benchmarks="hmmt_feb25:8",
    server_args="--mamba_ssm_cache_dtype float32 --no-enable-prefix-caching",
    output_dir=output_dir + model_name + "/no-python",
)