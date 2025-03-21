import random
from google import genai
from pydantic import BaseModel
import threading
from api_key import GEMINI_API_KEY

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Import the global lock if needed in this file
print_lock = threading.Lock()

class ResponseFormat(BaseModel):
    explanation_importance: str
    important_q_num: int
    explanation_attention: str
    attention_q_num: int

def _print_comparison_info(qa_pair1, qa_pair2, comparison_index):
    """Print initial comparison information."""
    with print_lock:
        print(f"Comparison {comparison_index + 1}: Comparing for Importance and Attention:") 
        print(f"QA Pair 1: UIN: {qa_pair1['uin']}, Heading: {qa_pair1['heading']}")
        print(f"QA Pair 2: UIN: {qa_pair2['uin']}, Heading: {qa_pair2['heading']}")
        print(f"Calling LLM for comparison {comparison_index + 1} - "
              f"Headings: '{qa_pair1['heading']}' vs '{qa_pair2['heading']}'...")

def _prepare_llm_prompt(qa_pair1, qa_pair2):
    """Prepare the prompt for the LLM."""
    MAX_LENGTH = 10_000
    return f"""These are parliamentary questions

    For Importance: Which is more important for the UK?
    explanation_importance: The explaination of which is more important 75 words max.
    important_q_num: The number either 1 or 2 which question is more important

    For Attention: Which issue is receiving enough focus in Westminster?
    explanation_attention: The explaination of which is receiving enough attention 75 words max.
    attention_q_num: The number either 1 or 2 which question is receiving enough focus

    --- 1 ---
    {qa_pair1['heading']}
    {qa_pair1['question_text'][:MAX_LENGTH]}
    {qa_pair1['answer_text'][:MAX_LENGTH]}
    --- 2 ---
    {qa_pair2['heading']}
    {qa_pair2['question_text'][:MAX_LENGTH]}
    {qa_pair2['answer_text'][:MAX_LENGTH]}
    """

def _get_llm_response(prompt):
    """Get response from LLM."""
    return gemini_client.models.generate_content(
        model='gemini-2.0-flash-lite',
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
            'response_schema': ResponseFormat,
            'system_instruction': 'Judge 2 UK Westminster debates: (1) importance, (2) under-focus. JSON.'
        }
    )

def _determine_winners(response, qa_pair1, qa_pair2):
    """Determine winners for importance and attention from LLM response."""
    # Handle importance
    if response.parsed.important_q_num == 1:
        winner_importance, loser_importance = qa_pair1, qa_pair2
    elif response.parsed.important_q_num == 2:
        winner_importance, loser_importance = qa_pair2, qa_pair1
    else:
        with print_lock:
            print("ERROR DECODING IMPORTANCE FROM LLM")
        return None, None, None, None

    # Handle attention
    if response.parsed.attention_q_num == 1:
        winner_attention, loser_attention = qa_pair1, qa_pair2
    elif response.parsed.attention_q_num == 2:
        winner_attention, loser_attention = qa_pair2, qa_pair1
    else:
        with print_lock:
            print("ERROR DECODING ATTENTION FROM LLM")
        return None, None, None, None

    return winner_importance, loser_importance, winner_attention, loser_attention

def _print_results(winner_importance, loser_importance, winner_attention, comparison_index):
    """Print the results of the comparison."""
    with print_lock:
        print(f"LLM Selection - Comparison {comparison_index + 1}:")
        print(f" More Important:   UIN: {winner_importance['uin']}, "
              f"Heading: {winner_importance['heading']}   (vs UIN: {loser_importance['uin']})")
        print(f" More Attention Grabbing: UIN: {winner_attention['uin']}, "
              f"Heading: {winner_attention['heading']} (vs UIN: {loser_importance['uin']})")

def eval_importance_attention(qa_pair1, qa_pair2, comparison_index):
    """
    Evaluates which of the two question/answer pairs is more important and which is receiving more 
    attention in Westminster. Uses a single LLM call to determine both importance and attention.
    """
    _print_comparison_info(qa_pair1, qa_pair2, comparison_index)
    
    prompt = _prepare_llm_prompt(qa_pair1, qa_pair2)
    response = _get_llm_response(prompt)
    
    with print_lock:
        print(response.text)
    
    winners = _determine_winners(response, qa_pair1, qa_pair2)
    if not all(winners):
        return None, None
        
    winner_importance, loser_importance, winner_attention, loser_attention = winners
    _print_results(winner_importance, loser_importance, winner_attention, comparison_index)

    return winner_importance, winner_attention


def update_elo_ratings(qa_pair1, qa_pair2, winner_importance, winner_attention):
    """Update Elo ratings based on comparison outcome for both importance and attention."""
    k_factor = 32  # Adjust K-factor as needed

    # Update Importance Elo Ratings
    rating1_importance = qa_pair1['elo_importance_rating']
    rating2_importance = qa_pair2['elo_importance_rating']

    probability_pair1_wins_importance = 1 / (1 + 10**((rating2_importance - rating1_importance) / 400))
    probability_pair2_wins_importance = 1 - probability_pair1_wins_importance

    with threading.Lock(): # Use a local lock here as qa_pairs_lock will be in main.py
        if winner_importance['id'] == qa_pair1['id']:  # Pair 1 wins on importance
            qa_pair1['elo_importance_rating'] = (rating1_importance + 
                k_factor * (1 - probability_pair1_wins_importance))
            qa_pair2['elo_importance_rating'] = (rating2_importance + 
                k_factor * (0 - probability_pair2_wins_importance))
        else:  # Pair 2 wins on importance
            qa_pair1['elo_importance_rating'] = (rating1_importance + 
                k_factor * (0 - probability_pair1_wins_importance))
            qa_pair2['elo_importance_rating'] = (rating2_importance + 
                k_factor * (1 - probability_pair2_wins_importance))

        # Update Attention Elo Ratings
        rating1_attention = qa_pair1['elo_attention_rating']
        rating2_attention = qa_pair2['elo_attention_rating']

        probability_pair1_wins_attention = 1 / (1 + 10**((rating2_attention - rating1_attention) / 400))
        probability_pair2_wins_attention = 1 - probability_pair1_wins_attention

        if winner_attention['id'] == qa_pair1['id']:  # Pair 1 wins on attention
            qa_pair1['elo_attention_rating'] = (rating1_attention + 
                k_factor * (1 - probability_pair1_wins_attention))
            qa_pair2['elo_attention_rating'] = (rating2_attention + 
                k_factor * (0 - probability_pair2_wins_attention))
        else:  # Pair 2 wins on attention
            qa_pair1['elo_attention_rating'] = (rating1_attention + 
                k_factor * (0 - probability_pair1_wins_attention))
            qa_pair2['elo_attention_rating'] = (rating2_attention + 
                k_factor * (1 - probability_pair2_wins_attention))


def initialize_elo_ratings(qa_pairs, initial_rating=1500):
    """Initialize Elo ratings for QA pairs, both for importance and attention."""
    for qa_pair in qa_pairs:
        qa_pair['elo_importance_rating'] = initial_rating
        qa_pair['elo_attention_rating'] = initial_rating
        qa_pair['percentile_importance_rank'] = 0 # Initialize percentile ranks
        qa_pair['percentile_attention_rank'] = 0   # Initialize percentile ranks
        qa_pair['unattended_score'] = 0 # Initialize unattended score


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
            diff_importance = abs(qa_pairs[i]['elo_importance_rating'] - 
                                qa_pairs[j]['elo_importance_rating'])
            diff_attention = abs(qa_pairs[i]['elo_attention_rating'] - 
                               qa_pairs[j]['elo_attention_rating'])
            elo_differences_importance.append(((qa_pairs[i], qa_pairs[j]), diff_importance))
            elo_differences_attention.append(((qa_pairs[i], qa_pairs[j]), diff_attention))

    if not elo_differences_importance: # No pairs to compare
        return None, None

    # Sort pairs by Elo difference
    elo_differences_importance.sort(key=lambda item: item[1])
    elo_differences_attention.sort(key=lambda item: item[1])

    # Define weighting factor that shifts focus to closer matches as comparisons increase
    convergence_factor = comparison_count / num_comparisons if num_comparisons > 0 else 0
    # Increase weight for closer matches as we do more comparisons
    close_match_probability_weight = convergence_factor 

    # Weighted random selection based on Elo difference for Importance
    if random.random() < close_match_probability_weight:
        # Favor closer matches - select from the lower end of sorted differences
        # Gradually reduce candidates from 50% to 10% as convergence_factor goes from 0 to 1
        num_candidates = max(1, int(len(elo_differences_importance) * 
                                  (0.5 - 0.4 * convergence_factor)))
        candidate_pairs_importance = elo_differences_importance[:num_candidates]
    else:
        # Explore wider range - select from all pairs
        candidate_pairs_importance = elo_differences_importance

    # Weighted random selection based on Elo difference for Attention
    if random.random() < close_match_probability_weight:
        num_candidates = max(1, int(len(elo_differences_attention) * 
                                  (0.5 - 0.4 * convergence_factor)))
        candidate_pairs_attention = elo_differences_attention[:num_candidates]
    else:
        candidate_pairs_attention = elo_differences_attention

    # Randomly choose a pair from the selected candidates (prioritizing by elo difference)
    pair_importance = random.choice(candidate_pairs_importance)[0]
    # Could use different pairs for attention but using same to save LLM calls
    pair_attention = random.choice(candidate_pairs_attention)[0]

    # Using importance pair for both to save LLM calls in this example
    return pair_importance[0], pair_importance[1]