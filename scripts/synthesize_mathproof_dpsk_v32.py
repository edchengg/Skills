from nemo_skills.pipeline.cli import generate, wrap_arguments, eval

cluster = "slurm"
output_dir = "/lustre/fsw/portfolios/llmservice/users/yachen/AceMath/Skills/deepseek-v32-sdg"
gpus=8
server_nodes=2
i=0
generate(
    ctx=wrap_arguments(
        "++skip_filled=True "
        "++prompt_config=gpt-oss/math_proof_gen "
        "++inference.top_p=0.95 "
        "++inference.tokens_to_generate=96000 " 
        "++max_concurrent_requests=1024 "
        "++inference.endpoint_type=chat "
        "++chat_template_kwargs.thinking=true "
    ),
    cluster=cluster,
    model="deepseek-ai/DeepSeek-V3.2-Speciale",
    with_sandbox=False,
    expname=f"dpsk_{i}",
    server_type='sglang',
    server_gpus=gpus,
    partition='batch',
    server_nodes=server_nodes,
    num_chunks=2,
    dependent_jobs=24,
    starting_seed=0,
    num_random_seeds=8,
    input_file="/lustre/fsw/portfolios/llmservice/projects/llmservice_fm_text/users/yachen/AceMath/AceProof/nemotron_math_proofs_v1_aops_min10000.jsonl",
    output_dir=f"{output_dir}/nemotron_math_proofs_v1_aops_min10000/",
    server_args=f"--ep-size {gpus * server_nodes} --dp {gpus * server_nodes} --enable-dp-attention --reasoning-parser deepseek-v3 --log-requests --mem-fraction-static=0.8",
)