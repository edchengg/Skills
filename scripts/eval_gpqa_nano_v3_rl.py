from nemo_skills.pipeline.cli import generate, wrap_arguments, eval

cluster = "slurm"
output_dir = "/lustre/fsw/portfolios/llmservice/users/yachen/AceMath/Skills/nano-v3-rl-eval/"
gpus=1
server_nodes=1
i=0
for step in [50, 100, 150, 200, 250]:
    model_name = f"nano-v3-rl-step-{step}"
    model_path = f"/lustre/fsw/portfolios/llmservice/projects/llmservice_fm_text/users/yachen/AceMath/nemo-rl-internal/results/nano_v3/step_{step}/huggingface"
    for benchmark in ["mmlu-pro"]:
        eval(
            ctx=wrap_arguments(
                "++inference.tokens_to_generate=120000 "
                "++inference.temperature=1.0 "
                "++inference.top_p=1.0 "
                "++prompt_config=eval/aai/mcq-10choices-boxed "
            ),
            cluster=cluster,
            expname=f"{model_name}-no-python",
            model=model_path,
            server_type='vllm',
            server_gpus=8,
            num_chunks=4,
            benchmarks=f"{benchmark}:1",
            server_args="--mamba_ssm_cache_dtype float32 --no-enable-prefix-caching",
            output_dir=output_dir + model_name + "/no-python-boxed",
        )

    model_name = f"nano-v3-rl-nosci-step-{step}"
    model_path = f"/lustre/fsw/portfolios/llmservice/projects/llmservice_fm_text/users/yachen/AceMath/nemo-rl-internal/results/nano_v3_noscience/step_{step}/huggingface"
    for benchmark in ["mmlu-pro"]:
        eval(
            ctx=wrap_arguments(
                "++inference.tokens_to_generate=120000 "
                "++inference.temperature=1.0 "
                "++inference.top_p=1.0 "
                "++prompt_config=eval/aai/mcq-10choices-boxed "
            ),
            cluster=cluster,
            expname=f"{model_name}-no-python",
            model=model_path,
            server_type='vllm',
            server_gpus=8,
            num_chunks=4,
            benchmarks=f"{benchmark}:1",
            server_args="--mamba_ssm_cache_dtype float32 --no-enable-prefix-caching",
            output_dir=output_dir + model_name + "/no-python-boxed",
        )