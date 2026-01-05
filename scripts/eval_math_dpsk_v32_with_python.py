from nemo_skills.pipeline.cli import generate, wrap_arguments, eval

cluster = "slurm"
output_dir = "/scratch/fsw/portfolios/llmservice/users/yachen/AceMath/Skills/deepseek-v32-eval"
gpus=8
server_nodes=2
i=0

server_args = (
    f"--ep-size {gpus * server_nodes} "
    f"--dp {gpus * server_nodes} "
    "--enable-dp-attention "
    "--reasoning-parser deepseek-v3 "
    "--log-requests "
    "--mem-fraction-static=0.8 "
    "--tool-call-parser deepseekv32 "
)

eval(
    ctx=wrap_arguments(
        "++skip_filled=True "
        "++prompt_config=qwen/math-tir "
        "++inference.top_p=0.95 "
        "++inference.tokens_to_generate=120000 " 
        "++max_concurrent_requests=1024 "
        "++inference.endpoint_type=chat "
        "++chat_template_kwargs.thinking=true "
        "++tool_modules=[nemo_skills.mcp.servers.python_tool::PythonTool] "
    ),
    cluster=cluster,
    model="/scratch/fsw/portfolios/llmservice/users/yachen/cache/DeepSeek-V3.2",
    server_container="/scratch/fsw/portfolios/llmservice/users/yachen/AceMath/container/nemo-skills-sglang-v32.sqsh",
    with_sandbox=True,
    expname=f"eval_{i}",
    server_type='sglang',
    server_gpus=gpus,
    partition='batch',
    server_nodes=server_nodes,
    num_chunks=1,
    dependent_jobs=0,
    benchmarks=f"imo_answerbench:1",
    output_dir=f"{output_dir}/with-python/",
    server_args=server_args,
)