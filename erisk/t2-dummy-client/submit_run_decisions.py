import requests
import json
import random
import time

# Base URL of the server
BASE_URL = "http://10.56.35.20:8080"  # Adjust to the actual server URL


# Function to load usernames from a .txt file
def load_user_nicks(file_path):
    """
    Reads a text file with one username per line and returns a set of usernames.
    
    :param file_path: Path to the .txt file containing usernames.
    :return: A set of usernames.
    """
    with open(file_path, "r") as file:
        # Read each line, strip whitespace, and add it to a set
        user_nicks = {line.strip() for line in file if line.strip()}
    return user_nicks


def submit_decisions(team_token, run, decisions, retries=5, backoff_factor=1):
    """
    Sends a POST request to submit team decisions for a specified team and run, with retry logic and exponential backoff.

    :param team_token: The unique token of the team.
    :param run: The run identifier.
    :param decisions: List of dictionaries representing TeamDecision objects.
    :param retries: Maximum number of retries (default is 5).
    :param backoff_factor: Backoff multiplier for wait time (default is 1 second).
    :return: JSON response containing persisted decisions or None if all retries fail.
    """
    endpoint_url = f"{BASE_URL}/submit/{team_token}/{run}"

    for attempt in range(retries):
        try:
            response = requests.post(endpoint_url, json=decisions, headers={"Content-Type": "application/json"})
            if response.status_code == 200:
                print(f"Decisions submitted successfully for team '{team_token}', run '{run}'.")
                return response.json()
            else:
                print(f"Failed to submit decisions. Status code: {response.status_code}. Attempt {attempt + 1} of {retries}.")
        except requests.exceptions.RequestException as e:
            print(f"Request error on attempt {attempt + 1} of {retries}: {e}")

        # Exponential backoff
        time.sleep(backoff_factor * (2 ** attempt))

    print(f"Failed to submit decisions after {retries} attempts.")
    return None

