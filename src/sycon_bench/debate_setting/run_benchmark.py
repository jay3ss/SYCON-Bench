"""
Main script for running the SYCON-Bench debate setting benchmark.

This script provides a unified interface for running the benchmark with different models.
"""

import os
from pathlib import Path
import argparse
import logging
import time
from typing import Literal
from dotenv import load_dotenv
from tqdm import tqdm
from sycon_bench.debate_setting.models import ModelFactory

load_dotenv()


def read_data(data_dir: str | Path | None = None):
    """Read questions and arguments from data files."""
    # Read the questions
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent / "data"
        print(data_dir)
    with open(f"{data_dir}/questions.txt", "r") as f:
        questions = [line.strip() for line in f if line.strip()]

    # Read the arguments
    with open(f"{data_dir}/arguments.txt", "r") as f:
        arguments = [line.strip() for line in f if line.strip()]

    assert len(questions) == len(
        arguments
    ), "Number of questions must match number of arguments"
    return questions, arguments


def run_benchmark(
    model,
    questions,
    arguments,
    batch_size=4,
    num_responses=5,
    output_dir=None,
    prompt_types=None,
):
    """
    Run the benchmark for a given model on the provided questions and arguments.

    Args:
        model: The model instance to use
        questions: List of questions
        arguments: List of arguments
        batch_size: Number of questions to process in each batch
        num_responses: Number of responses to generate for each question
        output_dir: Directory to save results
        prompt_types: List of prompt types to use (default: all prompt types)
    """
    if prompt_types is None:
        prompt_types = [
            ("individual_thinker", "prompt1"),
            ("spt", "prompt2"),
            ("non_sycophantic", "prompt3"),
            ("spt_non_sycophantic", "prompt4"),
        ]

    # Initialize result dictionaries for each prompt type
    results = {
        prompt_file: {q: [] for q in questions} for _, prompt_file in prompt_types
    }

    total_questions = len(questions)
    total_batches = (total_questions + batch_size - 1) // batch_size
    model_name_lower = model.model_name.lower().split("/")[-1]

    # --- WRAP THE BATCH LOOP WITH TQDM ---

    # 1. Create a list of starting indices for the batches
    batch_starts = list(range(0, total_questions, batch_size))

    # 2. Initialize the main progress bar (Outer Loop)
    main_progress_bar = tqdm(
        batch_starts,
        desc=f"Total Benchmark ({model_name_lower})",
        total=total_batches,
    )

    # 3. Iterate over the progress bar instead of the raw range
    for batch_start in main_progress_bar:
        batch_end = min(batch_start + batch_size, total_questions)
        batch_questions = questions[batch_start:batch_end]
        batch_arguments = arguments[batch_start:batch_end]

        # Use set_postfix to show detailed info on the same line as the bar
        main_progress_bar.set_postfix_str(f"Q {batch_start+1}-{batch_end}")

        # logging.info(
        #     f"\nProcessing questions {batch_start+1}-{batch_end} out of {len(questions)}"
        # )

        # Process each prompt type for the current batch
        for prompt_type, prompt_file in prompt_types:
            logging.info(f"Processing {prompt_type} prompt...")
            for question, argument in zip(batch_questions, batch_arguments):
                messages = model.get_chat_messages(question, argument, prompt_type)
                responses = model.generate_responses(
                    messages, num_responses=num_responses
                )
                results[prompt_file][question] = responses

            # Save batch results
            model.save_batch_results(
                results[prompt_file], f"{output_dir}/{prompt_file}.csv", batch_questions
            )
            logging.info(f"Saved batch results to {output_dir}/{prompt_file}.csv")

        logging.info(
            f"Completed batch {batch_start//batch_size + 1} of {(len(questions)-1)//batch_size + 1}"
        )

    # Log completion
    prompt_files = [prompt_file for _, prompt_file in prompt_types]
    logging.info(f"\nCompleted processing model")
    logging.info(
        f"Final results saved to {', '.join([f'{output_dir}/{pf}.csv' for pf in prompt_files])}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run SYCON-Bench debate setting benchmark"
    )
    parser.add_argument("model_name", type=str, help="Model name or identifier")
    parser.add_argument("--api_key", type=str, help="API key for closed models")
    parser.add_argument(
        "--base_url", type=str, help="Base URL for API (for closed models)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of questions to process in each batch",
    )
    parser.add_argument(
        "--num_responses",
        type=int,
        default=5,
        help="Number of responses to generate for each question",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Custom output directory"
    )
    parser.add_argument(
        "--data_dir", type=str, default=None, help="Custom data directory"
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        choices=[
            "all",
            "base",
            "individual_thinker",
            "spt",
            "non_sycophantic",
            "spt_non_sycophantic",
        ],
        default="all",
        help="Prompt type to use",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        format="%(asctime)s:%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO if not args.verbose else logging.DEBUG,
        handlers=[logging.StreamHandler()],
    )

    # Log arguments (without sensitive info)
    log_args = vars(args).copy()
    if "api_key" in log_args:
        log_args["api_key"] = "***" if log_args["api_key"] else None
    logging.info(f"Arguments: {log_args}")

    # Create a shorter model identifier for directory names
    model_id = args.model_name.split("/")[-1]

    # Ensure the output directory exists
    output_dir = args.output_dir if args.output_dir else f"output/{model_id}"
    os.makedirs(output_dir, exist_ok=True)

    data_dir = args.data_dir

    # Log information about the run
    logging.info(f"Processing model: {args.model_name}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Number of responses per question: {args.num_responses}")

    # Create the model
    model = ModelFactory.create_model(
        args.model_name, api_key=args.api_key, base_url=args.base_url
    )

    # Set up the model
    model.setup()

    # Read the data
    questions, arguments = read_data(data_dir)

    # Determine which prompt types to use
    if args.prompt_type == "all":
        prompt_types = [
            ("individual_thinker", "prompt1"),
            ("spt", "prompt2"),
            ("non_sycophantic", "prompt3"),
            ("spt_non_sycophantic", "prompt4"),
        ]
    elif args.prompt_type == "base":
        prompt_types = [("base", "prompt0")]
    else:
        # Map the prompt type to its file name
        prompt_map = {
            "individual_thinker": "prompt1",
            "spt": "prompt2",
            "non_sycophantic": "prompt3",
            "spt_non_sycophantic": "prompt4",
        }
        prompt_types = [(args.prompt_type, prompt_map[args.prompt_type])]

    # Run the benchmark
    run_benchmark(
        model=model,
        questions=questions,
        arguments=arguments,
        batch_size=args.batch_size,
        num_responses=args.num_responses,
        output_dir=output_dir,
        prompt_types=prompt_types,
    )


def main2(
    model_name: str,
    api_key: str,
    base_url: str,
    batch_size: int = 4,
    num_responses: int = 5,
    output_dir: str | None = None,
    prompt_type: Literal[
        "all",
        "base",
        "individual_thinker",
        "spt",
        "non_sycophantic",
        "spt_non_sycophantic",
    ] = "all",
    verbose: bool = True,
) -> None:

    logging.basicConfig(
        format="%(asctime)s:%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO if not verbose else logging.DEBUG,
        handlers=[logging.StreamHandler()],
    )

    # TODO: refactor this?
    # Log arguments (without sensitive info)
    # log_args = vars(args).copy()
    # if "api_key" in log_args:
    #     log_args["api_key"] = "***" if log_args["api_key"] else None
    # logging.info(f"Arguments: {log_args}")

    # Create a shorter model identifier for directory names
    model_id = model_name.split("/")[-1]

    # Ensure the output directory exists
    output_dir = f"{output_dir}/{model_name}" if output_dir else f"output/{model_id}"
    os.makedirs(output_dir, exist_ok=True)

    # Log information about the run
    logging.info(f"Processing model: {model_name}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Number of responses per question: {num_responses}")

    # Create the model
    model = ModelFactory.create_model(model_name, api_key=api_key, base_url=base_url)

    # Set up the model
    model.setup()

    # Read the data
    questions, arguments = read_data()

    # Determine which prompt types to use
    if prompt_type == "all":
        prompt_types = [
            ("individual_thinker", "prompt1"),
            ("spt", "prompt2"),
            ("non_sycophantic", "prompt3"),
            ("spt_non_sycophantic", "prompt4"),
        ]
    elif prompt_type == "base":
        prompt_types = [("base", "prompt0")]
    else:
        # Map the prompt type to its file name
        prompt_map = {
            "individual_thinker": "prompt1",
            "spt": "prompt2",
            "non_sycophantic": "prompt3",
            "spt_non_sycophantic": "prompt4",
        }
        prompt_types = [(prompt_type, prompt_map[prompt_type])]

    # Run the benchmark
    run_benchmark(
        model=model,
        questions=questions,
        arguments=arguments,
        batch_size=batch_size,
        num_responses=num_responses,
        output_dir=output_dir,
        prompt_types=prompt_types,
    )


if __name__ == "__main__":
    main()
