#!/usr/bin/env python3
"""
Relaunch only missing random seeds for acereasonv3_lowpassrate generation.

This script:
1. Checks which output-rs*.jsonl files already exist
2. Identifies missing seeds
3. Launches jobs only for missing seeds
"""
import os
import sys

# Configuration
OUTPUT_DIR = "/lustre/fsw/portfolios/llmservice/users/yachen/AceMath/Skills/gpt-oss-sdg/with-python/acereasonv3_lowpassrate"
INPUT_FILE = "/lustre/fsw/portfolios/llmservice/projects/llmservice_fm_text/users/yachen/AceMath/AceProof/data_processing_rl/acereason_v3_lowpassrate_data.jsonl"

def find_missing_seeds(output_dir, total_seeds=32):
    """Find which random seeds are missing."""
    os.chdir(output_dir)

    expected = set(range(total_seeds))
    existing = set()

    for i in range(total_seeds):
        filepath = f"output-rs{i}.jsonl"
        if os.path.exists(filepath):
            # Check if file is not empty
            size = os.path.getsize(filepath)
            if size > 100:  # At least 100 bytes to be considered valid
                existing.add(i)
                print(f"✓ Seed {i}: exists ({size:,} bytes)")
            else:
                print(f"✗ Seed {i}: exists but empty/invalid ({size} bytes)")

    missing = sorted(expected - existing)

    print(f"\n{'='*60}")
    print(f"Total seeds: {total_seeds}")
    print(f"Completed: {len(existing)}")
    print(f"Missing: {len(missing)}")
    print(f"{'='*60}")

    if missing:
        print(f"\nMissing seeds to relaunch: {missing}")
    else:
        print("\n✓ All seeds are complete!")

    return missing


def launch_seed(seed_number):
    """Launch a single random seed job."""
    cluster = "slurm"

    print(f"\nLaunching seed {seed_number}...")

    generate(
        ctx=wrap_arguments(
            "++inference.tokens_to_generate=90000 "
            "++inference.temperature=1.0 "
            "++inference.top_p=1.0 "
            "++prompt_config=gpt-oss/math "
            "++inference.endpoint_type=text "
            "++code_tags=gpt-oss "
            "++code_execution=true "
            "++server.code_execution.max_code_executions=100 "
            "++chat_template_kwargs.reasoning_effort=high "
            "++chat_template_kwargs.builtin_tools=[python] "
        ),
        cluster=cluster,
        expname=f"gpt-oss-sdg-math-with-python-rs{seed_number}",
        model="openai/gpt-oss-120b",
        server_type='vllm',
        server_gpus=8,
        input_file=INPUT_FILE,
        output_dir=OUTPUT_DIR,
        server_args="--async-scheduling",
        with_sandbox=True,
        num_chunks=1,
        dependent_jobs=0,
        starting_seed=seed_number,
        num_random_seeds=1,  # Only launch 1 seed at a time
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Relaunch missing random seeds for acereasonv3_lowpassrate'
    )
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check status without launching jobs'
    )
    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        help='Specific seeds to launch (e.g., --seeds 3 4 11). If not specified, all missing seeds will be launched.'
    )

    args = parser.parse_args()

    # Find missing seeds
    missing_seeds = find_missing_seeds(OUTPUT_DIR)

    if not missing_seeds:
        print("\n✓ All seeds completed successfully!")
        return

    if args.check_only:
        print("\n(Check-only mode: no jobs launched)")
        return

    # Determine which seeds to launch
    if args.seeds:
        seeds_to_launch = [s for s in args.seeds if s in missing_seeds]
        if not seeds_to_launch:
            print(f"\nError: Specified seeds {args.seeds} are not in missing seeds {missing_seeds}")
            return
        print(f"\nWill launch specified seeds: {seeds_to_launch}")
    else:
        seeds_to_launch = missing_seeds
        print(f"\nWill launch all missing seeds: {seeds_to_launch}")

    # Confirm before launching
    response = input(f"\nLaunch {len(seeds_to_launch)} job(s)? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Cancelled.")
        return

    # Launch jobs
    for seed in seeds_to_launch:
        try:
            launch_seed(seed)
            print(f"✓ Launched seed {seed}")
        except Exception as e:
            print(f"✗ Failed to launch seed {seed}: {e}")

    print(f"\n✓ Launched {len(seeds_to_launch)} job(s)")
    print("Use --check-only to verify status later")


if __name__ == '__main__':
    main()
