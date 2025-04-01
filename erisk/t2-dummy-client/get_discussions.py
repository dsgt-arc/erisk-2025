import time
import requests
import json
import os

# Base URL of the server
BASE_URL = "http://10.56.35.20:8080"  # Change this to your actual server URL

def get_discussions(team_token, retries=5, backoff_factor=1):
    """
    Sends a GET request to retrieve discussions for a team, with retry logic and exponential backoff.

    :param team_token: The unique token of the team.
    :param retries: Maximum number of retries (default is 5).
    :param backoff_factor: Backoff multiplier for wait time (default is 1 second).
    :return: JSON response containing discussions or None if all retries fail.
    """
    endpoint_url = f"{BASE_URL}/getdiscussions/{team_token}"

    for attempt in range(retries):
        try:
            response = requests.get(endpoint_url)
            if response.status_code == 200:
                print(f"Discussions retrieved successfully for team '{team_token}'.")
                return response.json()
            else:
                print(f"Failed to retrieve discussions. Status code: {response.status_code}. Attempt {attempt + 1} of {retries}.")
        except requests.exceptions.RequestException as e:
            print(f"Request error on attempt {attempt + 1} of {retries}: {e}")

        # Exponential backoff
        time.sleep(backoff_factor * (2 ** attempt))

    print(f"Failed to retrieve discussions after {retries} attempts.")
    return None
    

def get_list_target_subjects(data):
    """
    Extracts all 'targetSubject' values from the JSON data.

    :param data: JSON data, typically a list of submissions.
    :return: A list of unique 'targetSubject' values.
    """
    target_subjects = []

    for submission in data:
        # Check if 'targetSubject' exists in the submission and add it to the list
        if 'targetSubject' in submission:
            target_subjects.append(submission['targetSubject'])

    return target_subjects
