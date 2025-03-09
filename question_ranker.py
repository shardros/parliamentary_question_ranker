import requests
import requests_cache
import json
from datetime import date, timedelta
import random  # Import random for pair selection in Elo system
from google import genai
from pydantic import BaseModel
import time
import csv
import os  # Import os module for checkpoint file operations
import asyncio  # Import asyncio for async calls
from scipy.stats import percentileofscore  # Import scipy for percentile calculation
from api_key import GEMINI_API_KEY

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# --------------------------
# Get parlimentary questions

requests_cache.install_cache('parliament_api_cache', expire_after=None)

def fetch_question_by_id(question_id):
    """Fetch a specific written question by its ID."""
    url = f"https://questions-statements-api.parliament.uk/api/writtenquestions/questions/{question_id}"
    response = requests.get(url)
    if response.status_code == 200:
        try:
            question_json_string = response.text
            question_data = json.loads(question_json_string)
            return question_data
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error (ID {question_id}): {e}")
            return None
    else:
        print(f"API Error (ID {question_id}): {response.status_code}")
        return None


def fetch_answered_questions_ids_last_day(date_str_from, date_str_to, take=20):
    """Fetch IDs of answered written questions from the API for a date range."""
    url = "https://questions-statements-api.parliament.uk/api/writtenquestions/questions"
    params = {
        'answered': 'Answered',
        'answeredWhenFrom': date_str_from,
        'answeredWhenTo': date_str_to,
        'questionStatus': 'AnsweredOnly',
        'take': take
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        try:
            questions_json_string = response.text
            questions_data = json.loads(questions_json_string)
            return [item['value']['id'] for item in questions_data['results']]
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error (ID list): {e}")
            return []
    else:
        print(f"API Error (ID list): {response.status_code}")
        return []


def get_qa_pair_from_data(questions_data):
    """Extract question and answer text from question data."""
    if questions_data and 'value' in questions_data:
        value = questions_data['value']
        return {
            'id': value['id'],
            'uin': value['uin'],
            'heading': value['heading'],
            'question_text': value['questionText'],
            'answer_text': value['answerText']
        }
    return None

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
                'percentile_attention_rank': qa_pair['percentile_attention_rank']  # Save percentile ranks
            } for qa_pair in qa_pairs
        ],
        "comparison_count": comparison_count
    }
    try:
        with open(filename, 'w') as f:
            json.dump(checkpoint_data, f, indent=4)
        print(f"Checkpoint saved to {filename}")
    except Exception as e:
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
                    'percentile_importance_rank': qa_pair.get('percentile_importance_rank', 0), # Load percentile ranks, default to 0 if not found for backward compatibility
                    'percentile_attention_rank': qa_pair.get('percentile_attention_rank', 0)   # Load percentile ranks, default to 0 if not found for backward compatibility
                } for qa_pair in checkpoint_data['qa_pairs']
            ]
            comparison_count_loaded = checkpoint_data.get("comparison_count", 0) # Default to 0 if not found for backward compatibility
            print(f"Checkpoint loaded from {filename}, resuming from comparison {comparison_count_loaded}")
            return qa_pairs_loaded, comparison_count_loaded
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None, 0
    else:
        print("No checkpoint file found, starting from scratch.")
        return None, 0


# ------------------------------
# Rank questions

def print_ranked_questions_and_answers(ranked_qa_pairs_unattended): # Changed to ranked_qa_pairs_unattended
    """Print ranked question and answer pairs for unattended issues (most important, least attention)."""
    print("--- Ranked Question & Answer Pairs (Most Important, Least Attention - Last in List) ---") # Updated title
    for rank, qa_pair in enumerate(ranked_qa_pairs_unattended, start=1): # Changed to ranked_qa_pairs_unattended
        print(f"Rank {rank}: UIN: {qa_pair['uin']}, Heading: {qa_pair['heading']}, Unattended Score: {qa_pair['unattended_score']:.2f} (Importance Pct Rank: {qa_pair['percentile_importance_rank']:.2f}, Attention Pct Rank: {qa_pair['percentile_attention_rank']:.2f})") # Updated print statement to include percentile ranks and unattended score
        print(f"Question: {qa_pair['question_text']}")
        print(f"Answer: {qa_pair['answer_text']}")
        print("-" * 50)



async def eval_importance_attention(qa_pair1, qa_pair2, comparison_index):
    """
    Asynchronously evaluates which of the two question/answer pairs is more important and which is receiving more attention in Westminster.
    Uses a single LLM call to determine both importance and attention.
    """
    print(f"Comparison {comparison_index + 1}: Comparing for Importance and Attention:") # Added comparison index
    print(f"QA Pair 1: UIN: {qa_pair1['uin']}, Heading: {qa_pair1['heading']}")
    print(f"QA Pair 2: UIN: {qa_pair2['uin']}, Heading: {qa_pair2['heading']}")


    # Make sure to not use up all of our input tokens
    MAX_LENGTH = 10_000
    q1 = qa_pair1['question_text'][:MAX_LENGTH]
    a1 = qa_pair1['answer_text'][:MAX_LENGTH]
    q2 = qa_pair2['question_text'][:MAX_LENGTH]
    a2 = qa_pair2['answer_text'][:MAX_LENGTH]

    class ResponseFormat(BaseModel):
        explanation_importance: str
        important_q_num: int
        explanation_attention: str
        attention_q_num: int

    contents = f"""For the following two issues debated in westminster, judge which is more important for the UK and which is receiving more attention in Westminster.

    For Importance: Which is more important for the UK?
    explanation_importance: The explaination of which is more important 75 words max.
    important_q_num: The number either 1 or 2 which question is more important

    For Attention: Which issue is receiving more attention in Westminster?
    explanation_attention: The explaination of which is receiving more attention 75 words max.
    attention_q_num: The number either 1 or 2 which question is receiving more attention

    --- 1 ---
    {qa_pair1['heading']}
    {qa_pair1['question_text']}
    {qa_pair1['answer_text']}
    --- 2 ---
    {qa_pair2['heading']}
    {qa_pair2['question_text']}
    {qa_pair2['answer_text']}
    """

    response = await gemini_client.models.generate_content_async( # Made async
        model='gemini-2.0-flash-lite',
        contents=contents,
        config={
            'response_mime_type': 'application/json',
            'response_schema': ResponseFormat,
            'system_instruction': 'You are judging two issues debated in westminster. \
            For each issue you are judging (1) which is more important for the UK and (2) which is receiving more attention in westminster. Answer in JSON'
        }
    )
    print(response.text)
    if response.parsed.important_q_num == 1:
        winner_importance = qa_pair1
    elif response.parsed.important_q_num == 2:
        winner_importance = qa_pair2
    else:
        print("ERROR DECODING IMPORTANCE FROM LLM")
        return None, None

    if response.parsed.attention_q_num == 1:
        winner_attention = qa_pair1
    elif response.parsed.attention_q_num == 2:
        winner_attention = qa_pair2
    else:
        print("ERROR DECODING ATTENTION FROM LLM")
        return None, None

    print(f"Chosen as more important: UIN: {winner_importance['uin']}, Heading: {winner_importance['heading']}")
    print(f"Chosen as more attention grabbing: UIN: {winner_attention['uin']}, Heading: {winner_attention['heading']}")
    return winner_importance, winner_attention


def update_elo_ratings(qa_pair1, qa_pair2, winner_importance, winner_attention):
    """Update Elo ratings based on comparison outcome for both importance and attention."""
    k_factor = 32  # Adjust K-factor as needed

    # Update Importance Elo Ratings
    rating1_importance = qa_pair1['elo_importance_rating']
    rating2_importance = qa_pair2['elo_importance_rating']

    probability_pair1_wins_importance = 1 / (1 + 10**((rating2_importance - rating1_importance) / 400))
    probability_pair2_wins_importance = 1 - probability_pair1_wins_importance

    if winner_importance['id'] == qa_pair1['id']:  # Pair 1 wins on importance
        qa_pair1['elo_importance_rating'] = rating1_importance + k_factor * (1 - probability_pair1_wins_importance)
        qa_pair2['elo_importance_rating'] = rating2_importance + k_factor * (0 - probability_pair2_wins_importance)
    else:  # Pair 2 wins on importance
        qa_pair1['elo_importance_rating'] = rating1_importance + k_factor * (0 - probability_pair1_wins_importance)
        qa_pair2['elo_importance_rating'] = rating2_importance + k_factor * (1 - probability_pair2_wins_importance)

    # Update Attention Elo Ratings
    rating1_attention = qa_pair1['elo_attention_rating']
    rating2_attention = qa_pair2['elo_attention_rating']

    probability_pair1_wins_attention = 1 / (1 + 10**((rating2_attention - rating1_attention) / 400))
    probability_pair2_wins_attention = 1 - probability_pair1_wins_attention

    if winner_attention['id'] == qa_pair1['id']:  # Pair 1 wins on attention
        qa_pair1['elo_attention_rating'] = rating1_attention + k_factor * (1 - probability_pair1_wins_attention)
        qa_pair2['elo_attention_rating'] = rating2_attention + k_factor * (0 - probability_pair2_wins_attention)
    else:  # Pair 2 wins on attention
        qa_pair1['elo_attention_rating'] = rating1_attention + k_factor * (0 - probability_pair1_wins_attention)
        qa_pair2['elo_attention_rating'] = rating2_attention + k_factor * (1 - probability_pair2_wins_attention)


def initialize_elo_ratings(qa_pairs, initial_rating=1500):
    """Initialize Elo ratings for QA pairs, both for importance and attention."""
    for qa_pair in qa_pairs:
        qa_pair['elo_importance_rating'] = initial_rating
        qa_pair['elo_attention_rating'] = initial_rating
        qa_pair['percentile_importance_rank'] = 0 # Initialize percentile ranks
        qa_pair['percentile_attention_rank'] = 0  # Initialize percentile ranks
        qa_pair['unattended_score'] = 0 # Initialize unattended score


def calculate_percentile_ranks(qa_pairs):
    """Calculate percentile ranks for importance and attention Elo ratings."""
    importance_ratings = [qa_pair['elo_importance_rating'] for qa_pair in qa_pairs]
    attention_ratings = [qa_pair['elo_attention_rating'] for qa_pair in qa_pairs]

    for qa_pair in qa_pairs:
        qa_pair['percentile_importance_rank'] = percentileofscore(importance_ratings, qa_pair['elo_importance_rating'])
        qa_pair['percentile_attention_rank'] = percentileofscore(attention_ratings, qa_pair['elo_attention_rating'])

def calculate_unattended_score(qa_pairs):
    """Calculate unattended score based on percentile ranks (Attention - Importance)."""
    for qa_pair in qa_pairs:
        qa_pair['unattended_score'] = qa_pair['percentile_attention_rank'] - qa_pair['percentile_importance_rank'] # Attention - Importance for "most important/least attention LAST"


def rank_qa_pairs_unattended(qa_pairs): # New function to rank by unattended score
    """Rank QA pairs based on unattended score (percentile rank of attention - percentile rank of importance), in ascending order so most important/least attention is last.""" # Changed to ascending order
    return sorted(qa_pairs, key=lambda qa_pair: qa_pair['unattended_score'], reverse=False) # Sort in ascending order


def save_ranked_qa_to_csv(ranked_qa_pairs_unattended, filename=None): # Changed to ranked_qa_pairs_unattended
    """Save ranked question-answer pairs to a CSV file with timestamp, including unattended rank and score."""
    if filename is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"ranked_qa_pairs_unattended_{timestamp}.csv" # Updated filename

    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['rank_unattended', 'question', 'answer', 'elo_importance_rating', 'elo_attention_rating', 'percentile_importance_rank', 'percentile_attention_rank', 'unattended_score', 'question_id'] # Added percentile ranks and unattended score
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for rank_unattended, qa_pair in enumerate(ranked_qa_pairs_unattended, 1): # Changed to ranked_qa_pairs_unattended
                writer.writerow({
                    'rank_unattended': rank_unattended, # Changed to rank_unattended
                    'question': qa_pair['question_text'],
                    'answer': qa_pair['answer_text'],
                    'elo_importance_rating': qa_pair['elo_importance_rating'],
                    'elo_attention_rating': qa_pair['elo_attention_rating'],
                    'percentile_importance_rank': qa_pair['percentile_importance_rank'], # Save percentile ranks
                    'percentile_attention_rank': qa_pair['percentile_attention_rank'], # Save percentile ranks
                    'unattended_score': qa_pair['unattended_score'], # Save unattended score
                    'question_id': qa_pair['id']
                })
        print(f"Successfully saved ranked Q&A pairs to {filename}")
    except Exception as e:
        print(f"Error saving to CSV: {str(e)}")



def select_elo_based_pair(qa_pairs, comparison_count, num_comparisons):
    """
    Selects a pair of QA pairs for comparison, prioritizing pairs with closer Elo ratings
    as the comparison count increases, to speed up convergence.
    """
    if len(qa_pairs) < 2:
        return None, None

    # Calculate Elo rating differences for all pairs
    elo_differences_importance = []
    elo_differences_attention = []
    for i in range(len(qa_pairs)):
        for j in range(i + 1, len(qa_pairs)):
            diff_importance = abs(qa_pairs[i]['elo_importance_rating'] - qa_pairs[j]['elo_importance_rating'])
            diff_attention = abs(qa_pairs[i]['elo_attention_rating'] - qa_pairs[j]['elo_attention_rating'])
            elo_differences_importance.append(((qa_pairs[i], qa_pairs[j]), diff_importance))
            elo_differences_attention.append(((qa_pairs[i], qa_pairs[j]), diff_attention))

    if not elo_differences_importance: # No pairs to compare
        return None, None

    # Sort pairs by Elo difference
    elo_differences_importance.sort(key=lambda item: item[1])
    elo_differences_attention.sort(key=lambda item: item[1])

    # Define a weighting factor that shifts focus to closer matches as comparisons increase
    # This is a simple linear approach, can be tuned.
    convergence_factor = comparison_count / num_comparisons if num_comparisons > 0 else 0
    close_match_probability_weight = convergence_factor # Increase weight for closer matches as we do more comparisons


    # Weighted random selection based on Elo difference for Importance
    if random.random() < close_match_probability_weight:
        # Favor closer matches - select from the lower end of sorted differences
        num_candidates = max(1, int(len(elo_differences_importance) * (0.5 - 0.4 * convergence_factor))) # Gradually reduce candidates from 50% to 10% as convergence_factor goes from 0 to 1
        candidate_pairs_importance = elo_differences_importance[:num_candidates]
    else:
        # Explore wider range - select from all pairs
        candidate_pairs_importance = elo_differences_importance

    # Weighted random selection based on Elo difference for Attention - using same logic as importance for simplicity
    if random.random() < close_match_probability_weight:
        num_candidates = max(1, int(len(elo_differences_attention) * (0.5 - 0.4 * convergence_factor)))
        candidate_pairs_attention = elo_differences_attention[:num_candidates]
    else:
        candidate_pairs_attention = elo_differences_attention


    # Randomly choose a pair from the selected candidates (prioritizing by elo difference)
    pair_importance = random.choice(candidate_pairs_importance)[0]
    pair_attention = random.choice(candidate_pairs_attention)[0] # In practice importance and attention could use the same pairs to reduce LLM calls, but for clarity and potential future divergence, keeping separate for now.

    # For now, using importance pair for both evaluations to save on LLM calls in this example,
    # but could select pair_attention separately if needed for different comparison sets.
    return pair_importance[0], pair_importance[1] # Using importance pair for both to save LLM calls in this example. Could use pair_attention if needed.



async def get_answered_questions_last_day_elo_ranked(num_questions=20, num_comparisons=50, batch_size=20):  # Added num_comparisons and batch_size
    """Fetch, rank using Elo for importance and attention, and print Q&A for answered questions in the last day, using async LLM calls in batches."""
    today = date.today()
    n_days = 30
    end_date = today - timedelta(days=1)
    start_date = end_date - timedelta(days=n_days)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    # Load checkpoint if exists
    qa_pairs, comparison_count_start = load_checkpoint()
    if qa_pairs:
        print("Resuming from checkpoint...")
    else:
        comparison_count_start = 0
        question_ids = fetch_answered_questions_ids_last_day(start_date_str, end_date_str, take=num_questions)

        if not question_ids:
            date_range_str = f"{start_date_str} and {end_date_str}"
            print(f"No question IDs found for questions answered between {date_range_str}.")
            return

        qa_pairs = []
        print(f"Fetching details for {len(question_ids)} questions...")
        total_questions = len(question_ids)
        for i, question_id in enumerate(question_ids): # Added enumerate for progress bar
            question_data = fetch_question_by_id(question_id)
            if question_data:
                qa_pair = get_qa_pair_from_data(question_data)
                if qa_pair:
                    qa_pairs.append(qa_pair)
            else:
                print(f"Failed to fetch full data for question ID: {question_id}")

            # Progress bar implementation
            progress = (i + 1) / total_questions * 100
            print(f"Downloading questions: {progress:.2f}% ({i + 1}/{total_questions})", end='\r') # \r to overwrite the line

        print("\n") # New line after progress bar completes

        if not qa_pairs:
            print("No valid question/answer pairs fetched.")
            return

        initialize_elo_ratings(qa_pairs)  # Initialize Elo ratings for importance and attention
        save_checkpoint(qa_pairs, comparison_count_start) # Save initial state after fetching and initializing

    print(f"Performing {num_comparisons} comparisons in batches of {batch_size} to rank importance and attention...")  # Indicate comparisons

    comparison_tasks = [] # List to hold async tasks

    for comparison_count in range(comparison_count_start, num_comparisons): # Iterate through comparisons and track count
        pair1, pair2 = select_elo_based_pair(qa_pairs, comparison_count, num_comparisons) # Select pair based on Elo

        if pair1 is None or pair2 is None: # No more pairs to compare
            print("Not enough pairs left to compare.")
            break # Exit loop if no pairs are available

        comparison_tasks.append(eval_importance_attention(pair1, pair2, comparison_count)) # Append async task

        if len(comparison_tasks) >= batch_size or comparison_count == num_comparisons - 1: # Process batch or last comparison
            evaluation_results = await asyncio.gather(*comparison_tasks) # Run batch of tasks concurrently
            for task_index, (winner_pair_importance, winner_pair_attention) in enumerate(evaluation_results): # Process results
                pair1_batch, pair2_batch = select_elo_based_pair(qa_pairs, comparison_count_start + task_index, num_comparisons) # Re-select pairs - simplification, see comment in function
                if pair1_batch and pair2_batch and winner_pair_importance and winner_pair_attention: # Check pairs are valid before updating
                    update_elo_ratings(pair1_batch, pair2_batch, winner_pair_importance, winner_pair_attention)  # Update Elo ratings for both
            comparison_tasks = [] # Clear tasks for next batch
            save_checkpoint(qa_pairs, comparison_count + 1) # Save checkpoint after each batch

    calculate_percentile_ranks(qa_pairs) # Calculate percentile ranks for all QA pairs
    calculate_unattended_score(qa_pairs) # Calculate unattended score
    ranked_qa_pairs_unattended = rank_qa_pairs_unattended(qa_pairs)  # Rank based on Unattended Score (ascending) # Changed to ranked_qa_pairs_unattended

    print_ranked_questions_and_answers(ranked_qa_pairs_unattended)  # Print ranked list for unattended issues # Changed to ranked_qa_pairs_unattended

    # After ranking, save to CSV
    save_ranked_qa_to_csv(ranked_qa_pairs_unattended) # Changed to ranked_qa_pairs_unattended

    # Clean up checkpoint file after successful run
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print(f"Checkpoint file {CHECKPOINT_FILE} removed.")


if __name__ == "__main__":
    asyncio.run(get_answered_questions_last_day_elo_ranked(num_questions=1000, num_comparisons=5000, batch_size=100))  # Adjusted to pass num_comparisons and batch_size