import random
from datetime import date, timedelta
import time
import csv
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import percentileofscore
from api_key import GEMINI_API_KEY
from parliamentary_api import (
    fetch_answered_questions_ids_last_day,
    fetch_question_by_id,
    get_qa_pair_from_data,
    print_lock as parliament_print_lock  # Import the lock, though main.py will use its own
)
from importance import eval_importance_attention, ResponseFormat, update_elo_ratings, initialize_elo_ratings, select_elo_based_pair # Import from importance.py

# --------------------------
# Global lock for thread safety and print statements
qa_pairs_lock = threading.Lock() # Lock for qa_pairs access
print_lock = threading.Lock() # Lock for print statements

# ------------------------------
# Checkpoint functions

CHECKPOINT_FILE = "elo_ranking_checkpoint.json"

def save_checkpoint(qa_pairs, comparison_count, filename=CHECKPOINT_FILE):
    """Save the current state to a checkpoint file."""
    checkpoint_data = {
        "qa_pairs": [
            {
                'id': qa_pair['id'],
                'uin': qa_pair['uin'],
                'heading': qa_pair['heading'],
                'question_text': qa_pair['question_text'],
                'answer_text': qa_pair['answer_text'],
                'elo_importance_rating': qa_pair['elo_importance_rating'],
                'elo_attention_rating': qa_pair['elo_attention_rating'],
                'percentile_importance_rank': qa_pair['percentile_importance_rank'], # Save percentile ranks
                'percentile_attention_rank': qa_pair['percentile_attention_rank']   # Save percentile ranks
            } for qa_pair in qa_pairs
        ],
        "comparison_count": comparison_count
    }
    try:
        with open(filename, 'w') as f:
            json.dump(checkpoint_data, f, indent=4)
        with print_lock:
            print(f"Checkpoint saved to {filename}")
    except Exception as e:
        with print_lock:
            print(f"Error saving checkpoint: {e}")

def load_checkpoint(filename=CHECKPOINT_FILE):
    """Load state from a checkpoint file if it exists."""
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                checkpoint_data = json.load(f)
            qa_pairs_loaded = [
                {
                    'id': qa_pair['id'],
                    'uin': qa_pair['uin'],
                    'heading': qa_pair['heading'],
                    'question_text': qa_pair['question_text'],
                    'answer_text': qa_pair['answer_text'],
                    'elo_importance_rating': qa_pair['elo_importance_rating'],
                    'elo_attention_rating': qa_pair['elo_attention_rating'],
                    # Default to 0 if not found for backward compatibility
                    'percentile_importance_rank': qa_pair.get('percentile_importance_rank', 0),
                    'percentile_attention_rank': qa_pair.get('percentile_attention_rank', 0)
                } for qa_pair in checkpoint_data['qa_pairs']
            ]
            # Default to 0 if not found for backward compatibility
            comparison_count_loaded = checkpoint_data.get("comparison_count", 0)
            with print_lock:
                print(f"Checkpoint loaded from {filename}, "
                      f"resuming from comparison {comparison_count_loaded}")
            return qa_pairs_loaded, comparison_count_loaded
        except Exception as e:
            with print_lock:
                print(f"Error loading checkpoint: {e}")
            return None, 0
    else:
        with print_lock:
            print("No checkpoint file found, starting from scratch.")
        return None, 0


# ------------------------------
# Rank questions

def print_ranked_questions_and_answers(ranked_qa_pairs_unattended):
    """Print ranked Q&A pairs for unattended issues (most important, least attention)."""
    with print_lock:
        print("--- Ranked Q&A Pairs (Most Important, Least Attention - Last in List) ---")
        for rank, qa_pair in enumerate(ranked_qa_pairs_unattended, start=1):
            print(f"Rank {rank}: UIN: {qa_pair['uin']}, Heading: {qa_pair['heading']}, "
                  f"Unattended Score: {qa_pair['unattended_score']:.2f} "
                  f"(Importance Pct: {qa_pair['percentile_importance_rank']:.2f}, "
                  f"Attention Pct: {qa_pair['percentile_attention_rank']:.2f})")
            print(f"Question: {qa_pair['question_text']}")
            print(f"Answer: {qa_pair['answer_text']}")
            print("-" * 50)


def calculate_percentile_ranks(qa_pairs):
    """Calculate percentile ranks for importance and attention Elo ratings."""
    importance_ratings = [qa_pair['elo_importance_rating'] for qa_pair in qa_pairs]
    attention_ratings = [qa_pair['elo_attention_rating'] for qa_pair in qa_pairs]

    for qa_pair in qa_pairs:
        qa_pair['percentile_importance_rank'] = percentileofscore(
            importance_ratings, qa_pair['elo_importance_rating']
        )
        qa_pair['percentile_attention_rank'] = percentileofscore(
            attention_ratings, qa_pair['elo_attention_rating']
        )

def calculate_unattended_score(qa_pairs):
    """Calculate unattended score based on percentile ranks (Attention - Importance)."""
    for qa_pair in qa_pairs:
        # Attention - Importance for "most important/least attention LAST"
        qa_pair['unattended_score'] = (
            qa_pair['percentile_attention_rank'] - qa_pair['percentile_importance_rank']
        )


def rank_qa_pairs_unattended(qa_pairs):
    """Rank QA pairs based on unattended score in ascending order."""
    return sorted(qa_pairs, key=lambda qa_pair: qa_pair['unattended_score'], reverse=False)


def save_ranked_qa_to_csv(ranked_qa_pairs_unattended, filename=None):
    """Save ranked Q&A pairs to CSV with timestamp, including unattended rank and score."""
    if filename is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"ranked_qa_pairs_unattended_{timestamp}.csv"

    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'rank_unattended', 'question', 'answer', 'elo_importance_rating',
                'elo_attention_rating', 'percentile_importance_rank',
                'percentile_attention_rank', 'unattended_score', 'question_id'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for rank_unattended, qa_pair in enumerate(ranked_qa_pairs_unattended, 1):
                writer.writerow({
                    'rank_unattended': rank_unattended,
                    'question': qa_pair['question_text'],
                    'answer': qa_pair['answer_text'],
                    'elo_importance_rating': qa_pair['elo_importance_rating'],
                    'elo_attention_rating': qa_pair['elo_attention_rating'],
                    'percentile_importance_rank': qa_pair['percentile_importance_rank'],
                    'percentile_attention_rank': qa_pair['percentile_attention_rank'],
                    'unattended_score': qa_pair['unattended_score'],
                    'question_id': qa_pair['id']
                })
        with print_lock:
            print(f"Successfully saved ranked Q&A pairs to {filename}")
    except Exception as e:
        with print_lock:
            print(f"Error saving to CSV: {str(e)}")


def _get_date_range(n_days=30):
    """Get the start and end dates for fetching questions."""
    today = date.today()
    end_date = today - timedelta(days=1)
    start_date = end_date - timedelta(days=n_days)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    return start_date_str, end_date_str

def _load_or_fetch_qa_pairs(num_questions, start_date_str, end_date_str):
    """Loads QA pairs from checkpoint or fetches them."""
    qa_pairs, comparison_count_start = load_checkpoint()
    if qa_pairs:
        with print_lock:
            print("Resuming from checkpoint...")
        return qa_pairs, comparison_count_start
    else:
        comparison_count_start = 0
        question_ids = fetch_answered_questions_ids_last_day(
            start_date_str, end_date_str, take=num_questions
        )
        if not question_ids:
            date_range_str = f"{start_date_str} and {end_date_str}"
            with print_lock:
                print(f"No question IDs found for questions answered "
                      f"between {date_range_str}.")
            return 0

        qa_pairs = []
        with print_lock:
            print(f"Fetching details for {len(question_ids)} questions...")
        total_questions = len(question_ids)
        for i, question_id in enumerate(question_ids):
            question_data = fetch_question_by_id(question_id)
            if question_data:
                qa_pair = get_qa_pair_from_data(question_data)
                if qa_pair:
                    qa_pairs.append(qa_pair)
            else:
                with print_lock:
                    print(f"Failed to fetch full data for question ID: {question_id}")
            progress = (i + 1) / total_questions * 100
            with print_lock:
                print(f"Downloading questions: {progress:.2f}% "
                      f"({i + 1}/{total_questions})", end='\r')
        with print_lock:
            print("\n")
        if not qa_pairs:
            with print_lock:
                print("No valid question/answer pairs fetched.")
        return qa_pairs, comparison_count_start


def _perform_elo_comparisons(qa_pairs, comparison_count_start, num_comparisons, batch_size):
    """Performs ELO-based comparisons using threads."""
    with print_lock:
        print(f"Performing {num_comparisons} comparisons in batches "
              f"of {batch_size} to rank importance and attention...")

    comparison_tasks = []
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        for comparison_count in range(comparison_count_start, num_comparisons):
            pair1, pair2 = select_elo_based_pair(
                qa_pairs, comparison_count, num_comparisons
            )
            if pair1 is None or pair2 is None:
                with print_lock:
                    print("Not enough pairs left to compare.")
                break
            comparison_tasks.append(
                executor.submit(eval_importance_attention, pair1, pair2, comparison_count)
            )
            if len(comparison_tasks) >= batch_size or \
               comparison_count == num_comparisons - 1:
                evaluation_results = [task.result() for task in comparison_tasks]
                for task_index, (winner_pair_importance, winner_pair_attention) \
                        in enumerate(evaluation_results):
                    pair1_batch, pair2_batch = select_elo_based_pair(
                        qa_pairs, comparison_count_start + task_index, num_comparisons
                    )
                    if pair1_batch and pair2_batch and \
                       winner_pair_importance and winner_pair_attention:
                        with qa_pairs_lock:
                            update_elo_ratings(
                                pair1_batch, pair2_batch,
                                winner_pair_importance, winner_pair_attention
                            )
                comparison_tasks = []
                save_checkpoint(qa_pairs, comparison_count + 1)


def get_answered_questions_last_day_elo_ranked(num_questions=20, num_comparisons=50, batch_size=20):
    """Fetch, rank using Elo for importance and attention, and print Q&A."""
    start_date_str, end_date_str = _get_date_range()
    qa_pairs, comparison_count_start = _load_or_fetch_qa_pairs(
        num_questions, start_date_str, end_date_str
    )

    if not qa_pairs:
        return

    initialize_elo_ratings(qa_pairs)
    save_checkpoint(qa_pairs, comparison_count_start)

    _perform_elo_comparisons(
        qa_pairs, comparison_count_start, num_comparisons, batch_size
    )

    calculate_percentile_ranks(qa_pairs)
    calculate_unattended_score(qa_pairs)
    ranked_qa_pairs_unattended = rank_qa_pairs_unattended(qa_pairs)

    print_ranked_questions_and_answers(ranked_qa_pairs_unattended)
    save_ranked_qa_to_csv(ranked_qa_pairs_unattended)

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        with print_lock:
            print(f"Checkpoint file {CHECKPOINT_FILE} removed.")


if __name__ == "__main__":
    get_answered_questions_last_day_elo_ranked(num_questions=5, num_comparisons=50, batch_size=1)  # Adjusted to pass num_comparisons and batch_size