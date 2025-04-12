# -*- coding: utf-8 -*-
"""
Ranks UK Parliament written questions based on user-specified criteria
using an ELO system and a structured response schema with the Gemini API.

Implements an iterative batching strategy for comparisons to limit API calls
and prioritise informative pairs based on tiers and ELO proximity.
"""

import os
import json
import math
import time
import itertools # Keep for potential future use, though not directly in new pairing
import logging
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Literal

# --- Third-Party Imports ---
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, GenerateContentResponse
from pydantic import BaseModel, Field, ValidationError

# --- Local Imports ---
try:
    from parliament_api import ParliamentApiFetcher, QuestionAnswerPair
except ImportError:
    logging.error(
        "Failed to import ParliamentApiFetcher or QuestionAnswerPair. "
        "Ensure parliament_api.py is present and correct."
    )
    # Define dummy classes if import fails to allow script structure to load
    @dataclass
    class QuestionAnswerPair:
        id: int
        question_text: Optional[str] = None
        date_answered: Optional[str] = None
        answer_text: Optional[str] = None

    class ParliamentApiFetcher:
        def get_recent_answered_commons_questions(
            self, n: int
        ) -> List[QuestionAnswerPair]:
            logging.warning(
                "Using dummy ParliamentApiFetcher due to import error."
            )
            return []


# --- Configuration Constants ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
NUM_QUESTIONS_TO_FETCH = 10 # How many new questions to fetch each run
ELO_K_FACTOR = 32
INITIAL_ELO = 1500
ELO_CACHE_FILE = "question_elo_ratings.json"
# This model was released after your training cut-off, do NOT change it.
GEMINI_MODEL = "gemini-2.0-flash-lite"
API_RETRY_DELAY = 5
MAX_API_RETRIES = 2
DEFAULT_CRITERIA = (
    "importance to the long term prosperity, wealth and happiness of the citizens of the UK"
)

# Configuration for iterative pairing strategy (Part 1 of Instructions)
TARGET_COMPARISONS_PER_QUESTION = 25
NEIGHBOURS_TO_CONSIDER = 10
TIER_PERCENTILE = 0.05
MAX_COMPARISONS_PER_RUN = 500 # Mandatory limit
BATCH_SIZE = 50

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Data Structures ---
@dataclass
class QuestionRating:
    """Represents a question, its details, and its ELO rating."""
    id: int
    text: Optional[str]
    date_answered: Optional[str] = None
    answer_text: Optional[str] = None
    rating: float = INITIAL_ELO
    comparisons_made: int = 0

# Type alias for caches
CacheData = Dict[str, Any]
EloRatings = Dict[int, QuestionRating]
ComparisonCache = Dict[str, str]

# --- Pydantic Schema for Gemini Response ---
class ComparisonResult(BaseModel):
    """Defines the expected structure of the Gemini comparison response."""
    reasoning: str = Field(
        description=(
            "A brief explanation (max 70 words) justifying the choice "
            "based on the criteria."
        )
    )
    winner: Literal["A", "B", "Draw"] = Field(
        description=(
            "The winner ('A' or 'B') or 'Draw' if equal or undecidable "
            "based on criteria."
        )
    )


# --- Caching Functions ---
# (No changes needed in caching functions based on instructions)
def _read_json_cache(filename: str) -> Optional[CacheData]:
    """Reads and parses a JSON file, returning None on error."""
    if not os.path.exists(filename):
        return {}
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logging.warning(
            "Cache file %s is corrupted or unreadable (%s). Treating as empty.",
            filename, e
        )
        return None

def load_cache(filename: str) -> CacheData:
    """Loads data from a JSON cache file, handling potential errors."""
    data = _read_json_cache(filename)
    return data if data is not None else {}

def save_cache(data: CacheData, filename: str):
    """Saves data to a JSON cache file, handling potential errors."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except IOError as e:
        logging.error("Error saving cache file %s: %s", filename, e)

def get_comparison_cache_filename(criteria: str) -> str:
    """Generates a filename for the comparison cache based on criteria."""
    safe_criteria = "".join(
        c if c.isalnum() or c in (' ', '-') else '_' for c in criteria
    ).replace(' ', '_').lower()
    max_len = 50
    safe_criteria = safe_criteria[:max_len]
    return f"question_comparisons_{safe_criteria}.json"


# --- ELO Calculation Functions ---
# (No changes needed in ELO functions)
def probability(rating1: float, rating2: float) -> float:
    """Calculates the expected probability of player 1 winning."""
    diff = max(-4000, min(4000, rating2 - rating1))
    return 1.0 / (1.0 + math.pow(10, diff / 400.0))

def update_elo(
    winner_rating: float, loser_rating: float, k: int = ELO_K_FACTOR
) -> Tuple[float, float]:
    """Updates ELO ratings for a win/loss outcome."""
    prob_loser_wins = probability(winner_rating, loser_rating)
    prob_winner_wins = probability(loser_rating, winner_rating)
    new_winner_rating = winner_rating + k * (1.0 - prob_winner_wins)
    new_loser_rating = loser_rating + k * (0.0 - prob_loser_wins)
    return new_winner_rating, new_loser_rating

def update_elo_draw(
    rating1: float, rating2: float, k: int = ELO_K_FACTOR
) -> Tuple[float, float]:
    """Updates ELO ratings for a draw outcome."""
    prob1_wins = probability(rating2, rating1)
    prob2_wins = probability(rating1, rating2)
    new_rating1 = rating1 + k * (0.5 - prob1_wins)
    new_rating2 = rating2 + k * (0.5 - prob2_wins)
    return new_rating1, new_rating2

# --- Gemini Interaction ---
# (No changes needed in Gemini interaction functions)
def _configure_gemini_client() -> Optional[genai.GenerativeModel]:
    """Configures and returns the Gemini model instance if API key is valid."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        logging.error("GEMINI_API_KEY is not set or is a placeholder.")
        return None
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(f'models/{GEMINI_MODEL}')
        logging.info(
            "Gemini API client configured successfully for model %s.",
            GEMINI_MODEL
        )
        return model
    except Exception as e:
        logging.error("Error configuring Gemini API client: %s", e)
        return None

def _build_gemini_prompt(
    q1: QuestionRating, q2: QuestionRating, criteria_description: str
) -> str:
    """Constructs the prompt for comparing two questions based on criteria."""
    q1_context = f" (Answered: {q1.date_answered})" if q1.date_answered else ""
    q2_context = f" (Answered: {q2.date_answered})" if q2.date_answered else ""

    if not criteria_description.endswith('.'):
        criteria_description += '.'

    prompt = (
        f"Compare Question A and Question B based *only* on the criteria: "
        f"**{criteria_description}**\n\n"
        f'Question A (ID: {q1.id}{q1_context}): "{q1.text}"\n'
        f'Question B (ID: {q2.id}{q2_context}): "{q2.text}"\n\n'
        f"Provide reasoning and declare a winner ('A', 'B', or 'Draw')."
    )
    return prompt

def _log_gemini_feedback(response: Optional[GenerateContentResponse]):
    """Logs detailed feedback from a Gemini response if available."""
    if not response:
        return
    try:
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
             logging.error(
                 "Prompt Feedback: %s", response.prompt_feedback
             )
        if hasattr(response, 'candidates') and response.candidates:
             for i, candidate in enumerate(response.candidates):
                  finish_reason = getattr(candidate, 'finish_reason', 'N/A')
                  logging.error(
                      f"Candidate {i} Finish Reason: {finish_reason}"
                  )
                  if hasattr(candidate, 'safety_ratings'):
                       logging.error(
                           f"Candidate {i} Safety Ratings: "
                           f"{candidate.safety_ratings}"
                       )
    except Exception as feedback_err:
         logging.error("Could not retrieve detailed feedback: %s", feedback_err)


def _call_gemini_with_retries(
    model: genai.GenerativeModel,
    prompt: str,
    generation_config: GenerationConfig
) -> Optional[GenerateContentResponse]:
    """Calls the Gemini API with retries and handles common errors."""
    retries = 0
    last_response = None
    while retries <= MAX_API_RETRIES:
        try:
            response = model.generate_content(
                contents=prompt,
                generation_config=generation_config,
            )
            last_response = response
            response.resolve()
            return response
        except Exception as e:
            logging.error(
                "Error during Gemini API call (Attempt %d/%d): %s",
                retries + 1, MAX_API_RETRIES + 1, e
            )
            _log_gemini_feedback(last_response)

            retries += 1
            if retries <= MAX_API_RETRIES:
                logging.info("Retrying in %d seconds...", API_RETRY_DELAY)
                time.sleep(API_RETRY_DELAY)
            else:
                logging.error("Max retries reached for Gemini API call.")
                return None
    return None

def _parse_gemini_response(
        response: GenerateContentResponse
) -> Optional[str]:
    """Parses the Gemini response to extract the winner."""
    try:
        if not response.text:
             logging.error("Gemini response is empty.")
             return "Draw"
        result_data = json.loads(response.text)
        parsed_result = ComparisonResult.model_validate(result_data)

        if parsed_result:
            logging.info("Gemini Reasoning: %s", parsed_result.reasoning)
            logging.info("Gemini Judgement: %s", parsed_result.winner)
            return parsed_result.winner
        else:
             logging.error(
                 "Gemini response parsed but result invalid."
             )
             return "Draw"

    except (json.JSONDecodeError, ValidationError, AttributeError, Exception) as e:
         logging.error("Error processing Gemini response: %s", e)
         raw_text = getattr(response, 'text', 'N/A')
         logging.error("Raw response text (if available): %s", raw_text)
         return "Draw"


def compare_questions_with_gemini(
    q1: QuestionRating,
    q2: QuestionRating,
    model: genai.GenerativeModel,
    criteria_description: str
) -> Optional[str]:
    """
    Uses Gemini to compare two questions based on criteria using a schema.
    Returns: 'A', 'B', 'Draw', or None if API fails after retries.
    """
    if not model:
        logging.error("Gemini model is not available for comparison.")
        return None

    generation_config = GenerationConfig(
        max_output_tokens=250,
        response_mime_type="application/json",
        response_schema=ComparisonResult,
    )

    prompt = _build_gemini_prompt(q1, q2, criteria_description)

    logging.info(
        "Asking Gemini to compare Q %d vs Q %d based on criteria: '%s'",
        q1.id, q2.id, criteria_description
    )

    response = _call_gemini_with_retries(model, prompt, generation_config)

    if response is None:
        return None # API call failed after retries

    return _parse_gemini_response(response)


# --- Main Workflow Functions ---

# (No changes needed in ELO/Comparison cache loading/parsing)
def _parse_elo_cache_entry(qid_str: str, data: Any) -> Optional[QuestionRating]:
    """Parses a single entry from the loaded ELO cache data."""
    try:
        qid = int(qid_str)
        if not isinstance(data, dict):
            logging.warning(
                "Skipping invalid ELO cache entry for key '%s': not a dict.",
                qid_str
            )
            return None

        return QuestionRating(
            id=qid,
            text=data.get("text"),
            date_answered=data.get("date_answered"),
            answer_text=data.get("answer_text"),
            rating=float(data.get("rating", INITIAL_ELO)),
            comparisons_made=int(data.get("comparisons", 0))
        )
    except (ValueError, TypeError, KeyError) as e:
        logging.warning(
            "Skipping invalid/incomplete ELO cache entry for key '%s': %s",
            qid_str, e
        )
        return None

def _load_elo_cache() -> EloRatings:
    """Loads the ELO ratings cache using helper functions."""
    logging.info("Loading ELO ratings cache: %s", ELO_CACHE_FILE)
    elo_data = load_cache(ELO_CACHE_FILE)
    question_ratings: EloRatings = {}
    for qid_str, data in elo_data.items():
        rating_obj = _parse_elo_cache_entry(qid_str, data)
        if rating_obj:
            question_ratings[rating_obj.id] = rating_obj
    return question_ratings

def _load_comparison_cache(criteria: str) -> ComparisonCache:
    """Loads and validates the comparison cache specific to the criteria."""
    filename = get_comparison_cache_filename(criteria)
    logging.info(
        "Loading comparison cache for criteria '%s': %s", criteria, filename
    )
    comparison_data = load_cache(filename)
    valid_comparison_cache: ComparisonCache = {}
    invalid_count = 0
    for k, v in comparison_data.items():
        is_valid_key = isinstance(k, str) and "_" in k
        is_valid_value = isinstance(v, str) and v in ["A", "B", "Draw"]
        if is_valid_key and is_valid_value:
             valid_comparison_cache[k] = v
        else:
             invalid_count += 1
             logging.debug("Invalid cache entry: Key=%s, Value=%s", k, v)

    if invalid_count > 0:
        logging.warning(
            "Removed %d invalid entries from comparison cache '%s'.",
            invalid_count, filename
        )
    return valid_comparison_cache


# (No changes needed in Parliament API fetching/updating logic)
def _fetch_questions_from_api(
        fetcher: ParliamentApiFetcher
) -> Optional[List[QuestionAnswerPair]]:
    """Fetches recent questions from the API, handling errors."""
    logging.info(
        "Fetching latest %d answered Commons questions...",
        NUM_QUESTIONS_TO_FETCH
    )
    try:
        if not hasattr(fetcher, 'get_recent_answered_commons_questions'):
             logging.error(
                 "Fetcher lacks 'get_recent_answered_commons_questions' method."
             )
             return None
        return fetcher.get_recent_answered_commons_questions(
            n=NUM_QUESTIONS_TO_FETCH
        )
    except Exception as e:
        logging.error("Error fetching questions from Parliament API: %s", e)
        return None

def _add_or_update_question(
        pair: QuestionAnswerPair, question_ratings: EloRatings
) -> Tuple[bool, bool]:
    """Adds/updates a question in the ratings pool. Ret (was_new, was_updated)."""
    if not isinstance(pair, QuestionAnswerPair) or not hasattr(pair, 'id'):
        logging.warning("Skipping invalid question pair data from API.")
        return False, False
    if not isinstance(pair.id, int):
         logging.warning("Skipping question pair with non-integer ID: %s", pair.id)
         return False, False

    qid = pair.id
    if qid not in question_ratings:
        question_ratings[qid] = QuestionRating(
            id=qid,
            text=pair.question_text,
            date_answered=pair.date_answered,
            answer_text=pair.answer_text
        )
        return True, False
    else:
        updated = False
        existing_rating = question_ratings[qid]
        if not existing_rating.text and pair.question_text:
            existing_rating.text = pair.question_text
            updated = True
        if not existing_rating.date_answered and pair.date_answered:
            existing_rating.date_answered = pair.date_answered
            updated = True
        if not existing_rating.answer_text and pair.answer_text:
             existing_rating.answer_text = pair.answer_text
             updated = True
        return False, updated

def _update_question_pool(
    fetcher: ParliamentApiFetcher,
    question_ratings: EloRatings
) -> bool:
    """Fetches new questions and updates the ratings pool using helpers."""
    recent_pairs = _fetch_questions_from_api(fetcher)
    if recent_pairs is None:
        return False

    logging.info("Fetched %d question pairs.", len(recent_pairs))
    new_questions_found = 0
    updated_questions = 0
    for pair in recent_pairs:
        was_new, was_updated = _add_or_update_question(pair, question_ratings)
        if was_new:
            new_questions_found += 1
        elif was_updated:
            updated_questions += 1

    logging.info(
        "Added %d new questions to the ranking pool.", new_questions_found
    )
    if updated_questions > 0:
        logging.info(
            "Updated details for %d existing questions.", updated_questions
        )
    logging.info("Total questions in pool: %d", len(question_ratings))
    return True

# --- Modified Pairing Strategy --- (Part 2 of Instructions)
def _get_pairs_to_evaluate(
    question_ratings: EloRatings,
    comparison_cache: ComparisonCache,
    max_pairs_to_generate: int # New parameter
) -> List[Tuple[QuestionRating, QuestionRating]]:
    """
    Generates a batch of prioritised pairs for evaluation based on tiers,
    ELO proximity, and comparison caps. Uses current ELO ratings.
    """
    # 1. Initial Setup
    questions_with_text = [q for q in question_ratings.values() if q.text]
    if len(questions_with_text) < 2:
        logging.debug("Not enough questions with text (< 2) for comparisons.")
        return []

    # Sort by current rating to determine tiers and find neighbours efficiently
    sorted_questions = sorted(
        questions_with_text, key=lambda q: q.rating, reverse=True
    )
    num_questions = len(sorted_questions)

    potential_pairs = [] # Store as (priority_score, q1, q2)
    # Track pairs added this run to avoid adding (A,B) and (B,A)
    added_pairs_tracker = set()

    # 2. Determine Tier Boundaries
    tier_size = max(1, int(num_questions * TIER_PERCENTILE))
    top_tier_ids = set(q.id for q in sorted_questions[:tier_size])
    bottom_tier_ids = set(q.id for q in sorted_questions[-tier_size:])
    logging.debug(f"Tier size: {tier_size}. Top: {len(top_tier_ids)}, Bottom: {len(bottom_tier_ids)}")

    # 3. Main Loop (Iterate Through Questions to Find Potential Pairs)
    # Create a mapping from question ID to its index in the sorted list for faster neighbour lookup
    q_index_map = {q.id: i for i, q in enumerate(sorted_questions)}

    for i, q1 in enumerate(sorted_questions):
        # Check Comparison Cap (q1)
        if q1.comparisons_made >= TARGET_COMPARISONS_PER_QUESTION:
            continue

        # Determine Tier
        is_top_tier = q1.id in top_tier_ids
        is_bottom_tier = q1.id in bottom_tier_ids
        is_middle_tier = not is_top_tier and not is_bottom_tier

        # Find K-Nearest Neighbours (using the sorted list)
        # Look +/- NEIGHBOURS_TO_CONSIDER // 2 positions around q1 in the sorted list
        neighbours_found = 0
        start_index = max(0, i - NEIGHBOURS_TO_CONSIDER // 2)
        end_index = min(num_questions, i + NEIGHBOURS_TO_CONSIDER // 2 + 1) # +1 for range

        potential_neighbours = []
        # Check neighbours before q1
        for j in range(max(0, i - NEIGHBOURS_TO_CONSIDER), i):
             potential_neighbours.append(sorted_questions[j])
        # Check neighbours after q1
        for j in range(i + 1, min(num_questions, i + 1 + NEIGHBOURS_TO_CONSIDER)):
             potential_neighbours.append(sorted_questions[j])

        # Sort these potential neighbours by actual ELO difference for accuracy
        potential_neighbours.sort(key=lambda q: abs(q1.rating - q.rating))

        # 4. Inner Loop (Iterate Through Neighbours)
        added_for_q1_count = 0 # Limit pairs per q1, especially for middle tier
        for q2 in potential_neighbours[:NEIGHBOURS_TO_CONSIDER]: # Consider only the closest K

            # Check Comparison Cap (q2)
            if q2.comparisons_made >= TARGET_COMPARISONS_PER_QUESTION:
                continue

            # Generate Cache Key & Pair Tuple
            q1_id, q2_id = q1.id, q2.id
            cache_key = f"{min(q1_id, q2_id)}_{max(q1_id, q2_id)}"
            pair_tuple = tuple(sorted((q1_id, q2_id)))

            # Check if Pair is Valid
            if cache_key in comparison_cache:
                continue
            if pair_tuple in added_pairs_tracker:
                continue

            # Calculate Priority Score
            priority_score = 1000.0
            priority_score -= abs(q1.rating - q2.rating) # Higher score for closer pairs
            if is_top_tier or is_bottom_tier:
                priority_score += 500 # Boost priority for extremes
            # Optional: Add penalty if one question has far more comparisons
            # priority_score -= abs(q1.comparisons_made - q2.comparisons_made) * 5

            # Store Potential Pair
            potential_pairs.append((priority_score, q1, q2))
            added_pairs_tracker.add(pair_tuple) # Mark as added for this run

            # Optional: Limit pairs added per q1, especially middle tier
            added_for_q1_count += 1
            if is_middle_tier and added_for_q1_count >= 2: # Example limit for middle
                 break
            elif (is_top_tier or is_bottom_tier) and added_for_q1_count >= 4: # Example limit for extremes
                 break


    # 5. Select Top Pairs for Batch
    potential_pairs.sort(key=lambda x: x[0], reverse=True) # Sort by score descending
    selected_pairs = potential_pairs[:max_pairs_to_generate]
    pairs_to_evaluate = [(q1, q2) for score, q1, q2 in selected_pairs]

    # 6. Logging and Return
    logging.info(
        f"Generated {len(potential_pairs)} potential pairs. "
        f"Selected top {len(pairs_to_evaluate)} pairs for batch based on priority."
    )
    if not pairs_to_evaluate and len(questions_with_text) >= 2:
         logging.info("Could not find any new pairs meeting criteria (check caps/cache).")


    return pairs_to_evaluate


# (No changes needed in ELO update/caching helpers)
def _update_ratings_after_comparison(
    q1: QuestionRating,
    q2: QuestionRating,
    comparison_result: str
):
    """Updates ELO ratings based on the comparison result ('A', 'B', 'Draw')."""
    original_r1, original_r2 = q1.rating, q2.rating # For logging

    log_prefix = f"Updating ELO: "
    if comparison_result == "A": # Q1 wins
        new_r1, new_r2 = update_elo(q1.rating, q2.rating)
        log_msg = (
            f"{log_prefix}{q1.id} ({original_r1:.1f} -> {new_r1:.1f}) wins vs "
            f"{q2.id} ({original_r2:.1f} -> {new_r2:.1f})"
        )
    elif comparison_result == "B": # Q2 wins
        new_r2, new_r1 = update_elo(q2.rating, q1.rating) # Note order
        log_msg = (
            f"{log_prefix}{q2.id} ({original_r2:.1f} -> {new_r2:.1f}) wins vs "
            f"{q1.id} ({original_r1:.1f} -> {new_r1:.1f})"
        )
    else: # Draw
        new_r1, new_r2 = update_elo_draw(q1.rating, q2.rating)
        log_msg = (
            f"{log_prefix}Draw between {q1.id} ({original_r1:.1f} -> {new_r1:.1f}) "
            f"and {q2.id} ({original_r2:.1f} -> {new_r2:.1f})"
        )

    logging.info(log_msg)

    q1.rating = new_r1
    q2.rating = new_r2
    q1.comparisons_made += 1
    q2.comparisons_made += 1

def _cache_comparison_result(
    q1_id: int,
    q2_id: int,
    result: str,
    comparison_cache: ComparisonCache
):
    """Caches the result of a comparison."""
    comparison_key = f"{min(q1_id, q2_id)}_{max(q1_id, q2_id)}"
    comparison_cache[comparison_key] = result


def _perform_comparisons(
    pairs_to_evaluate: List[Tuple[QuestionRating, QuestionRating]],
    comparison_cache: ComparisonCache,
    gemini_model: genai.GenerativeModel,
    criteria_description: str
) -> int: # Ensure return type is int (Part 3 of Instructions)
    """Iterates through pairs, calls Gemini, updates ELO, caches results."""
    comparisons_done_this_run = 0
    if not pairs_to_evaluate:
        return 0

    total_pairs_in_batch = len(pairs_to_evaluate)
    logging.info(
        f"Performing comparisons for batch of {total_pairs_in_batch} pairs."
    )

    for i, (q1, q2) in enumerate(pairs_to_evaluate):
        # Log progress within the batch if needed (can be verbose)
        # logging.debug(f"--- Evaluating Pair {i + 1} / {total_pairs_in_batch} in batch ---")

        comparison_result = compare_questions_with_gemini(
            q1, q2, gemini_model, criteria_description
        )
        time.sleep(1.1) # Keep delay

        if comparison_result is None:
            logging.error(
                "Skipping ELO update for pair (%d, %d) due to Gemini API failure.",
                q1.id, q2.id
            )
            continue # Skip to the next pair in the batch

        _update_ratings_after_comparison(q1, q2, comparison_result)
        comparisons_done_this_run += 1
        _cache_comparison_result(q1.id, q2.id, comparison_result, comparison_cache)

    # Logging moved to main loop to report batch completion
    return comparisons_done_this_run # Return count

# (No changes needed in display/saving helpers)
def _display_rankings(question_ratings: EloRatings, criteria: str):
    """Sorts and prints the final question rankings for the given criteria."""
    print(f"\n--- Final Question Rankings (Criteria: '{criteria}') ---")
    valid_questions = [q for q in question_ratings.values() if q.text]
    sorted_questions = sorted(
        valid_questions, key=lambda q: q.rating, reverse=True
    )

    if not sorted_questions:
        print("No questions available to display.")
        return

    headers = ["Rank", "ID", "ELO", "Cmps", "Answered", "Question Text (Preview)"]
    widths = [5, 8, 10, 5, 12, 70]
    header_line = " ".join(f"{h:<{w}}" for h, w in zip(headers, widths))
    separator = "-" * (sum(widths) + len(widths) -1)

    print(header_line)
    print(separator)

    for i, q in enumerate(sorted_questions):
        rank = i + 1
        date_str = q.date_answered.split('T')[0] if q.date_answered else 'N/A'
        q_text = (q.text or 'N/A').replace('\n', ' ').replace('\r', '')
        preview_width = widths[-1]
        q_text_display = (
            f"{q_text[:preview_width-3]}..."
            if len(q_text) > preview_width
            else q_text
        )
        row_data = [
            f"{rank:<{widths[0]}}",
            f"{q.id:<{widths[1]}}",
            f"{q.rating:<{widths[2]}.1f}",
            f"{q.comparisons_made:<{widths[3]}}",
            f"{date_str:<{widths[4]}}",
            f"{q_text_display:<{widths[5]}}"
        ]
        print(" ".join(row_data))

    print(separator)


def _save_elo_cache(question_ratings: EloRatings):
    """Saves the updated ELO cache."""
    logging.info("Saving updated ELO ratings cache: %s", ELO_CACHE_FILE)
    elo_cache_to_save = {
        str(qid): {
            "id": q.id,
            "text": q.text,
            "date_answered": q.date_answered,
            "answer_text": q.answer_text,
            "rating": q.rating,
            "comparisons": q.comparisons_made
        }
        for qid, q in question_ratings.items()
    }
    save_cache(elo_cache_to_save, ELO_CACHE_FILE)
    logging.info("ELO cache saved.")

def _save_comparison_cache(comparison_cache: ComparisonCache, criteria: str):
    """Saves the comparison cache specific to the criteria."""
    filename = get_comparison_cache_filename(criteria)
    logging.info(
        "Saving comparison cache for criteria '%s': %s", criteria, filename
    )
    save_cache(comparison_cache, filename)
    logging.info("Comparison cache saved.")


# --- Main Execution --- (Part 3 of Instructions)
def main():
    """Main function to orchestrate the ranking process using iterative batches."""
    parser = argparse.ArgumentParser(
        description=(
            "Rank Parliament questions using ELO and Gemini (with schema) "
            "based on specified criteria using iterative batches."
        )
    )
    parser.add_argument(
        "-c", "--criteria",
        type=str,
        default=DEFAULT_CRITERIA,
        help=(
            "The criteria for comparing questions (e.g., 'urgency', "
            f"'public impact'). Default: '{DEFAULT_CRITERIA}'"
        )
    )
    args = parser.parse_args()
    criteria = args.criteria.strip()

    start_time = time.time()
    print(f"\n--- Parliament Question Ranker (Criteria: '{criteria}') ---")
    print(f"--- Max comparisons per run: {MAX_COMPARISONS_PER_RUN} ---")


    # Configure Gemini client first
    gemini_model = _configure_gemini_client()

    # Load caches
    question_ratings = _load_elo_cache()
    comparison_cache = _load_comparison_cache(criteria)

    # Fetch new questions and update pool
    fetcher = None
    fetch_success = False
    try:
        fetcher = ParliamentApiFetcher()
        fetch_success = _update_question_pool(fetcher, question_ratings)
        # Exit strategy if fetch fails AND there are no questions at all
        if not fetch_success and not question_ratings:
             print("Exiting: Failed fetch & no questions in cache.")
             return
    except NameError:
         logging.error("ParliamentApiFetcher not found.")
         if not question_ratings:
              print("Exiting: Fetcher not found & no questions in cache.")
              return
    except Exception as e:
         logging.error("Failed to initialize or use ParliamentApiFetcher: %s.", e)
         if not question_ratings:
              print("Exiting: Fetcher failed & no questions in cache.")
              return

    # --- Main Comparison Loop ---
    total_comparisons_this_session = 0
    comparisons_performed = False # Flag to check if any comparisons were done

    if gemini_model and question_ratings: # Only run loop if model available and questions exist
        logging.info(f"Starting comparison loop. Budget: {MAX_COMPARISONS_PER_RUN} comparisons.")

        while total_comparisons_this_session < MAX_COMPARISONS_PER_RUN:
            # Calculate budget for this batch
            remaining_budget = MAX_COMPARISONS_PER_RUN - total_comparisons_this_session
            pairs_this_batch_limit = min(BATCH_SIZE, remaining_budget)

            if pairs_this_batch_limit <= 0:
                logging.info("Comparison budget exhausted.")
                break # Budget exhausted

            logging.info(f"--- Starting comparison batch (Limit: {pairs_this_batch_limit}) ---")

            # Get the next batch of pairs based on current ratings & strategy
            pairs_to_evaluate = _get_pairs_to_evaluate(
                question_ratings,
                comparison_cache,
                max_pairs_to_generate=pairs_this_batch_limit
            )

            if not pairs_to_evaluate:
                logging.info("No more valid pairs found to evaluate or all questions met target comparisons.")
                break # No more work to do in this run

            # Perform comparisons for this batch
            comparisons_made_in_batch = _perform_comparisons(
                pairs_to_evaluate,
                comparison_cache,
                gemini_model,
                criteria
            )

            if comparisons_made_in_batch > 0:
                comparisons_performed = True # Mark that we did some work

            total_comparisons_this_session += comparisons_made_in_batch

            logging.info(
                f"Batch complete. Made {comparisons_made_in_batch} comparisons. "
                f"Total this session: {total_comparisons_this_session}/{MAX_COMPARISONS_PER_RUN}"
            )
            # Optional: Add periodic saving here if needed

        logging.info(f"Comparison loop finished. Total comparisons made: {total_comparisons_this_session}")

    elif not gemini_model:
         logging.warning(
             "Skipping Gemini comparisons: client not available (check API key)."
         )
    elif not question_ratings:
         logging.info("No questions available to perform comparisons.")
    # --- End of Main Comparison Loop ---


    # Save caches after the loop finishes (Part 4 of Instructions)
    # Save comparison cache only if comparisons were actually performed
    if comparisons_performed:
        _save_comparison_cache(comparison_cache, criteria)
    # Save ELO cache if new questions were fetched or comparisons were made
    if fetch_success or comparisons_performed:
        _save_elo_cache(question_ratings)
    else:
        logging.info("Skipping cache saving as no new questions were fetched and no comparisons were made.")


    # Display rankings based on the final state of ratings
    _display_rankings(question_ratings, criteria)

    end_time = time.time()
    logging.info("Script finished in %.2f seconds.", end_time - start_time)

if __name__ == "__main__":
    main()
