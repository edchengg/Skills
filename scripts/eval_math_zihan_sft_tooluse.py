"""
Evaluation script for models trained with JSON tool call format:
<tool_call>
{"name": "python", "arguments": {"code": "..."}}
</tool_call>
<tool_response>
output
</tool_response>

This uses the code_execution approach with custom JSON extraction.
"""

from nemo_skills.pipeline.cli import eval, wrap_arguments

cluster = "slurm"
output_dir = "/lustre/fsw/portfolios/llmservice/users/yachen/AceMath/Skills/zihan-sft-eval/"
model_name = "zihan-sft-29916"
model_path = "/lustre/fsw/portfolios/llmservice/users/zihanl/inform/megatron2hf/llm_ft/Post-Training/megatron-lm/checkpoints/sft_gptoss_v2_1_32nodes_allpurpose_5e-5_32_262144/safetensors-checkpoint-29916"

for benchmark in ["aime25", "hmmt_feb25"]:
    eval(
        ctx=wrap_arguments(
            "++inference.tokens_to_generate=120000 "
            "++inference.temperature=1.0 "
            "++inference.top_p=1.0 "
            "++prompt_config=generic/math_sft_tool "
            # Use text completions API for code execution
            "++inference.endpoint_type=text "
            # Use the custom code_tags for JSON tool call format
            "++code_tags=tool_call_json "
            # Enable code execution
            "++code_execution=true "
            "++server.code_execution.max_code_executions=100 "
            "++server.code_execution.code_execution_timeout=120 "
        ),
        cluster=cluster,
        expname=f"{model_name}-with-python-json",
        model=model_path,
        server_type='vllm',
        server_gpus=1,
        num_chunks=1,
        with_sandbox=True,
        benchmarks=f"{benchmark}:32",
        server_args="--mamba_ssm_cache_dtype float32 --no-enable-prefix-caching",
        output_dir=output_dir + model_name + "/with-python",
        # Use custom generation task that handles JSON tool calls
        generation_module="nemo_skills.inference.generate_tool_call_json",
    )

for benchmark in ["imo_answerbench"]:
    eval(
        ctx=wrap_arguments(
            "++inference.tokens_to_generate=120000 "
            "++inference.temperature=1.0 "
            "++inference.top_p=1.0 "
            "++prompt_config=generic/math_sft_tool "
            # Use text completions API for code execution
            "++inference.endpoint_type=text "
            # Use the custom code_tags for JSON tool call format
            "++code_tags=tool_call_json "
            # Enable code execution
            "++code_execution=true "
            "++server.code_execution.max_code_executions=100 "
            "++server.code_execution.code_execution_timeout=120 "
        ),
        cluster=cluster,
        expname=f"{model_name}-with-python-json",
        model=model_path,
        server_type='vllm',
        server_gpus=1,
        num_chunks=4,
        with_sandbox=True,
        benchmarks=f"{benchmark}:4",
        server_args="--mamba_ssm_cache_dtype float32 --no-enable-prefix-caching",
        output_dir=output_dir + model_name + "/with-python",
        # Use custom generation task that handles JSON tool calls
        generation_module="nemo_skills.inference.generate_tool_call_json",
    )