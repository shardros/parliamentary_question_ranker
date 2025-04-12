import requests
import time
import os # Keep os for path operations if pathlib isn't strictly required everywhere
import json
import atexit
from pathlib import Path # Use pathlib for path operations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# --- Data Structure ---
@dataclass
class QuestionAnswerPair:
    """Represents a simplified question-answer pair from the UK Parliament API."""
    id: int
    date_answered: Optional[str] = None
    question_text: Optional[str] = None
    answer_text: Optional[str] = None

# --- Fetcher Class ---
class ParliamentApiFetcher:
    """
    Fetches answered written questions from the UK Parliament API
    with local caching for question details. Refactored for clarity.
    """
    def __init__(self,
                 base_url: str = "https://writtenquestions-api.parliament.uk",
                 cache_filename: str = "parliament_question_cache.json",
                 rate_limit_delay: float = 0.1):
        """
        Initializes the fetcher, loads the cache, and registers saving.
        """
        self.base_url = base_url
        self.cache_path = Path(cache_filename) # Use pathlib.Path
        self.rate_limit_delay = rate_limit_delay
        self._cache: Dict[int, Dict[str, Any]] = {}
        self._load_cache()
        atexit.register(self._save_cache)

    def _load_cache(self):
        """Loads the question details cache from a JSON file if it exists."""
        if self.cache_path.exists():
            try:
                with self.cache_path.open('r') as f:
                    loaded_data = json.load(f)
                    # Ensure keys are integers after loading from JSON
                    self._cache = {int(k): v for k, v in loaded_data.items()}
                    print(f"Loaded {len(self._cache)} items from cache file: {self.cache_path}")
            except (json.JSONDecodeError, IOError, ValueError) as e:
                print(f"Warning: Could not load cache file {self.cache_path}. Starting with empty cache. Error: {e}")
                self._cache = {}
        else:
            print("Cache file not found. Starting with empty cache.")

    def _save_cache(self):
        """Saves the question details cache to a JSON file."""
        try:
            with self.cache_path.open('w') as f:
                json.dump(self._cache, f, indent=4)
                print(f"Saved {len(self._cache)} items to cache file: {self.cache_path}")
        except IOError as e:
            print(f"Warning: Could not save cache file {self.cache_path}. Error: {e}")

    def _fetch_question_details(self, question_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetches (or retrieves from cache) full details for a single question ID.
        """
        if question_id in self._cache:
            print(f"Cache hit for question ID: {question_id}")
            return self._cache[question_id]

        print(f"Cache miss for question ID: {question_id}. Fetching from API...")
        endpoint = f"{self.base_url}/api/writtenquestions/questions/{question_id}"
        try:
            time.sleep(self.rate_limit_delay) # Apply delay only on API call
            response = requests.get(endpoint, timeout=30)
            response.raise_for_status() # Check for HTTP errors
            data = response.json()

            if isinstance(data.get('value'), dict):
                details = data['value']
                self._cache[question_id] = details # Cache successful result
                return details
            else:
                print(f"Warning: Unexpected API response format for question ID {question_id}. Details: {data}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Warning: Could not fetch details for question ID {question_id}. Error: {e}")
            return None
        except Exception as e: # Catch other potential errors like JSON parsing
            print(f"Warning: An unexpected error occurred fetching details for question ID {question_id}. Error: {e}")
            return None

    def _fetch_question_list(self, params: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Fetches the list of question summaries based on provided parameters."""
        endpoint = f"{self.base_url}/api/writtenquestions/questions"
        print(f"\nFetching question list with params: {params}...")
        try:
            response = requests.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            list_data = response.json()

            if isinstance(list_data.get('results'), list):
                return list_data['results']
            else:
                print("Error: Unexpected API response format for question list.")
                print(f"Response: {list_data}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error fetching initial question list from Parliament API: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during initial list fetch: {e}")
            return None

    def _create_qa_pair(self, summary_data: Dict[str, Any], details_data: Optional[Dict[str, Any]]) -> Optional[QuestionAnswerPair]:
        """Creates a QuestionAnswerPair object from summary and detail data."""
        question_id = summary_data.get('id')
        if not isinstance(question_id, int):
            print(f"Warning: Invalid or missing question ID in summary data: {summary_data}")
            return None

        # Prioritize details_data if available
        q_text = details_data.get('questionText') if details_data else summary_data.get('questionText')
        a_text = details_data.get('answerText') if details_data else None
        date_ans = details_data.get('dateAnswered') if details_data else summary_data.get('dateAnswered')

        try:
            return QuestionAnswerPair(
                id=question_id,
                date_answered=date_ans,
                question_text=q_text,
                answer_text=a_text
            )
        except TypeError as e:
            # Should be rare if ID check passes, but good practice
            print(f"Warning: Could not create QuestionAnswerPair for ID {question_id}. Error: {e}.")
            return None

    def get_recent_answered_commons_questions(self, n: int) -> List[QuestionAnswerPair]:
        """
        Fetches the N most recent answered Commons questions as QuestionAnswerPair objects.

        Orchestrates fetching the list, then fetching details (using cache) for each item,
        and finally creating the result objects.
        """
        params = {
            'take': n,
            'answered': 'Answered',
            'house': 'Commons'
        }
        fetched_pairs: List[QuestionAnswerPair] = []

        # 1. Fetch the initial list of question summaries
        question_list_results = self._fetch_question_list(params)

        if question_list_results is None:
            return fetched_pairs # Return empty list if list fetch failed

        print(f"Processing {len(question_list_results)} questions from list...")
        # 2. Iterate, fetch details (cached), and create pairs
        for item in question_list_results:
            summary_data = item.get('value')
            if not isinstance(summary_data, dict):
                print(f"Warning: Skipping invalid item in question list (missing 'value' dict): {item}")
                continue

            question_id = summary_data.get('id')
            if not isinstance(question_id, int):
                 print(f"Warning: Skipping item with invalid ID in summary: {summary_data}")
                 continue

            # Fetch details (uses cache)
            details_data = self._fetch_question_details(question_id)

            # Create the pair object using summary and details
            qa_pair = self._create_qa_pair(summary_data, details_data)
            if qa_pair:
                fetched_pairs.append(qa_pair)

        return fetched_pairs

# --- Example Usage ---
if __name__ == "__main__":
    num_questions_to_fetch = 3

    print(f"\n--- Running Parliament Question Fetcher ---")
    # Create an instance of the fetcher
    fetcher = ParliamentApiFetcher()

    print(f"Attempting to fetch the {num_questions_to_fetch} most recent answered Commons questions...")
    # Call the main method
    recent_pairs_list = fetcher.get_recent_answered_commons_questions(num_questions_to_fetch)

    if recent_pairs_list:
        print(f"\nSuccessfully processed {len(recent_pairs_list)} question-answer pairs:")
        for i, qa_pair in enumerate(recent_pairs_list):
            print(f"\n--- Pair {i+1} (ID: {qa_pair.id}) ---")
            print(f"  Date Answered: {qa_pair.date_answered or 'N/A'}")
            question = qa_pair.question_text or "N/A"
            print(f"  Question Text: {question}")
            answer = qa_pair.answer_text or "N/A"
            print(f"  Answer Text: {answer}")
    else:
        print("\nFailed to fetch question pairs or no pairs found.")

    print("\n--- Fetcher Run Complete ---")
    # Cache saving happens automatically on exit

