import requests
import requests_cache
import json
import os
import time
from datetime import date, timedelta
import threading

# --------------------------
# Global lock for print statements used in this file
print_lock = threading.Lock() # Lock for print statements

# --------------------------
# Get parliamentary questions

requests_cache.install_cache('parliament_api_cache', expire_after=None)

LOCAL_QUESTIONS_DIR = "parliament_questions_local"

def ensure_local_questions_dir_exists():
    """Ensure the local questions directory exists."""
    if not os.path.exists(LOCAL_QUESTIONS_DIR):
        os.makedirs(LOCAL_QUESTIONS_DIR)

ensure_local_questions_dir_exists()

def save_question_locally(question_id, question_data):
    """Save question data to a local JSON file."""
    filepath = os.path.join(LOCAL_QUESTIONS_DIR, f"question_{question_id}.json")
    try:
        with open(filepath, 'w') as f:
            json.dump(question_data, f, indent=4)
        with print_lock:
            print(f"Question ID {question_id} saved locally to {filepath}")
        return True
    except Exception as e:
        with print_lock:
            print(f"Error saving question ID {question_id} locally: {e}")
        return False

def load_question_locally(question_id):
    """Load question data from a local JSON file."""
    filepath = os.path.join(LOCAL_QUESTIONS_DIR, f"question_{question_id}.json")
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                question_data = json.load(f)
            with print_lock:
                print(f"Question ID {question_id} loaded from local file {filepath}")
            return question_data
        except json.JSONDecodeError as e:
            with print_lock:
                print(f"JSON Decode Error loading local file for ID {question_id}: {e}")
            return None
        except Exception as e:
            with print_lock:
                print(f"Error loading local file for question ID {question_id}: {e}")
            return None
    return None


def fetch_question_by_id(question_id):
    """Fetch a specific written question by its ID, checking local cache first."""
    local_data = load_question_locally(question_id)
    if local_data:
        return local_data

    # Rate limit to 10 calls per second
    time.sleep(0.1)  # 100ms delay between calls

    url = f"https://questions-statements-api.parliament.uk/api/writtenquestions/questions/{question_id}"
    response = requests.get(url)
    if response.status_code == 200:
        try:
            question_json_string = response.text
            question_data = json.loads(question_json_string)
            save_question_locally(question_id, question_data) # Save question locally
            return question_data
        except json.JSONDecodeError as e:
            with print_lock:
                print(f"JSON Decode Error (ID {question_id}): {e}")
            return None
    else:
        with print_lock:
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
            with print_lock:
                print(f"JSON Decode Error (ID list): {e}")
            return
    else:
        with print_lock:
            print(f"API Error (ID list): {response.status_code}")
        return


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