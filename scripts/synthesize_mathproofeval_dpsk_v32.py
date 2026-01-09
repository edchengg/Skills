from nemo_skills.pipeline.cli import generate, wrap_arguments, eval

cluster = "slurm"
output_dir = "/scratch/fsw/portfolios/llmservice/users/yachen/AceMath/Skills/deepseek-v32-sdg"
gpus=8
server_nodes=1
i=0
generate(
    ctx=wrap_arguments(
        "++skip_filled=True "
        "++prompt_config=dpsk/math_proof_eval "
        "++inference.top_p=0.95 "
        "++inference.tokens_to_generate=96000 " 
        "++max_concurrent_requests=1024 "
        "++inference.endpoint_type=chat "
        "++chat_template_kwargs.thinking=true "
    ),
    cluster=cluster,
    model="/scratch/fsw/portfolios/llmservice/users/yachen/cache/DeepSeek-V3.2-Speciale",
    server_container="/scratch/fsw/portfolios/llmservice/users/yachen/AceMath/container/nemo-skills-sglang-v32.sqsh",
    with_sandbox=False,
    expname=f"generate_{i}",
    server_type='sglang',
    server_gpus=gpus,
    partition='batch',
    server_nodes=server_nodes,
    num_chunks=16,
    dependent_jobs=2,
    starting_seed=0,
    num_random_seeds=1,
    input_file="/scratch/fsw/portfolios/llmservice/users/yachen/AceMath/aceproof/data_processing_sft/data/proofs/math_proof_eval_input_0109/problem_proof.jsonl",
    output_dir=f"{output_dir}/nemotron_math_proofs_v1_aops_max6000tok_problem_proofs_0109/",
    server_args=f"--ep-size {gpus * server_nodes} --dp {gpus * server_nodes} --enable-dp-attention --reasoning-parser deepseek-v3 --log-requests --mem-fraction-static=0.8",
)

# generate(
#     ctx=wrap_arguments(
#         "++skip_filled=True "
#         "++prompt_config=dpsk/math_proof_eval "
#         "++inference.top_p=0.95 "
#         "++inference.tokens_to_generate=96000 " 
#         "++max_concurrent_requests=1024 "
#         "++inference.endpoint_type=chat "
#         "++chat_template_kwargs.thinking=true "
#     ),
#     cluster=cluster,
#     model="/scratch/fsw/portfolios/llmservice/users/yachen/cache/DeepSeek-V3.2-Speciale",
#     server_container="/scratch/fsw/portfolios/llmservice/users/yachen/AceMath/container/nemo-skills-sglang-v32.sqsh",
#     with_sandbox=False,
#     expname=f"generate_{i}",
#     server_type='sglang',
#     server_gpus=gpus,
#     partition='batch',
#     server_nodes=server_nodes,
#     num_chunks=10,
#     dependent_jobs=1,
#     starting_seed=0,
#     num_random_seeds=1,
#     input_file="/scratch/fsw/portfolios/llmservice/users/yachen/AceMath/aceproof/data_processing_sft/data/proofs/math_proof_eval_input/math_proof_eval_max6000tok_split2.jsonl",
#     output_dir=f"{output_dir}/nemotron_math_proofs_v1_aops_max6000tok_problem_proofs_split2/",
#     server_args=f"--ep-size {gpus * server_nodes} --dp {gpus * server_nodes} --enable-dp-attention --reasoning-parser deepseek-v3 --log-requests --mem-fraction-static=0.8",
# )