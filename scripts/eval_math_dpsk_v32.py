from nemo_skills.pipeline.cli import generate, wrap_arguments, eval
import json

loader_cfg = {
    "enable_multithread_load": True,
    "num_threads": 32,   # try 8/16 first; 110 is usually counterproductive on Lustre
}

cluster = "slurm"
output_dir = "/scratch/fsw/portfolios/llmservice/users/yachen/AceMath/Skills/deepseek-v32-eval"
gpus=8
server_nodes=2
i=0
eval(
    ctx=wrap_arguments(
        "++skip_filled=True "
        "++prompt_config=qwen/math-cot "
        "++inference.top_p=0.95 "
        "++inference.tokens_to_generate=120000 " 
        "++max_concurrent_requests=1024 "
        "++inference.endpoint_type=chat "
        "++chat_template_kwargs.thinking=true "
    ),
    cluster=cluster,
    model="deepseek-ai/DeepSeek-V3.2-Speciale",
    with_sandbox=False,
    expname=f"generate_{i}",
    server_type='sglang',
    server_gpus=gpus,
    partition='batch',
    server_nodes=server_nodes,
    num_chunks=1,
    dependent_jobs=0,
    benchmarks=f"aime25:1",
    output_dir=f"{output_dir}/",
    server_args=f"--ep-size {gpus * server_nodes} --dp {gpus * server_nodes} --enable-dp-attention --reasoning-parser deepseek-v3 --log-requests --mem-fraction-static=0.8 --model-loader-extra-config '{json.dumps(loader_cfg)}'",
)