from nemo_skills.pipeline.cli import generate, wrap_arguments, eval

cluster = "slurm"
output_dir = "/lustre/fsw/portfolios/llmservice/users/yachen/AceMath/Skills/zihan-sft-passrate/"
gpus=8
server_nodes=1
i=0
model_name = "zihan-sft-10992"
model_path = "/lustre/fsw/portfolios/llmservice/users/zihanl/inform/megatron2hf/llm_ft/Post-Training/megatron-lm/checkpoints/sft_gptoss_v1_1_32nodes_allpurpose_5e-5_32_262144/safetensors-checkpoint-10992"


# generate(
#     ctx=wrap_arguments(
#         "++inference.tokens_to_generate=120000 "
#         "++inference.temperature=1.0 "
#         "++inference.top_p=1.0 "
#         "++prompt_config=generic/math_sft_notool "
#     ),
#     cluster=cluster,
#     expname=f"{model_name}-no-python",
#     model=model_path,
#     server_type='vllm',
#     num_chunks=1,
#     server_args="--mamba_ssm_cache_dtype float32 --no-enable-prefix-caching",
#     output_dir=output_dir + model_name + "/acereasonv2",
#     # can customize the number of GPUs used
#     server_gpus=8,
#     input_file="/lustre/fsw/portfolios/llmservice/projects/llmservice_fm_text/users/yachen/AceMath/verl/examples/data_generation/data_final/acereason_2_data/acereason_v2_math_release.jsonl",
#     dependent_jobs=4,
#     starting_seed=0,
#     num_random_seeds=16,
# )

generate(
    ctx=wrap_arguments(
        # we are using fewer tokens than max context length as code output isn't accounted for
        "++inference.tokens_to_generate=96000 "
        # recommended inference settings including prompt config
        "++inference.temperature=1.0 "
        "++inference.top_p=1.0 "
        "++prompt_config=judge/imo-answerbench "
        "++inference.endpoint_type=text "
        "++code_tags=gpt-oss "
        "++chat_template_kwargs.reasoning_effort=high "
    ),
    generation_type="math_judge",
    model="openai/gpt-oss-120b",
    cluster=cluster,
    expname=f"{model_name}-judge",
    model=model_path,
    server_type='vllm',
    num_chunks=1,
    # can customize the number of GPUs used
    server_gpus=8,
    input_dir=output_dir + model_name + "/acereasonv2",
    output_dir=output_dir + model_name + "/acereasonv2-judge",
    dependent_jobs=1,
    starting_seed=0,
    num_random_seeds=1,
)
