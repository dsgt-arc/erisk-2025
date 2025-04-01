# eRisk Dummy Client

This repository contains a dummy Python client designed to interact with the eRisk REST service for demonstration purposes. The client provides functionalities for retrieving discussions and submitting decisions for specific team tokens and multiple runs.


---

## Repository Structure

```plaintext
├── get_discussions.py          # Script to retrieve discussions for a team
├── submit_run_decisions.py     # Script to submit team decisions for a specific run
├── dummy_client.py             # Main script that orchestrates the retrieval and submission processes
├── output_discussions/         # Directory to save discussion responses as JSON
├── dummy_users/                # Directory to save target subjects and dummy user lists
└── README.md                   # This README file
```

## Automated Round Processing

The dummy_client.py script automates the process of retrieving discussions and submitting decisions for the max number of runs (5).

### How it works:

    - Run-based submissions:
       1. Decisions are submitted for all runs sequentially before retrieving discussions.
       2. The number_of_runs parameter specifies how many runs (range from 0 to (number_of_runs - 1)) 
       to include in the submission process. The maximum value for number_of_runs is 5.
    - For each round:
       1. Submits decisions for all runs using submit_run_decisions.py.
       2. Retrieves discussions for the current round using get_discussions.py.


## Output Structure

    - Discussions
       1. Saved as JSON files in the output_discussions/ directory.
       2. Example file: output_discussions/<team_token>_discussions_number_<discussion_number>.json
    
    - Target Subjects
       1. Saved as a .txt file in the dummy_users/ directory.
       2. Example file: Example file: dummy_users/<team_token>_target_subjects.txt.


## Notes
    
    - Retry Logic
       1. All requests include retry limitations with exponential backoff to avoid saturating the server.
       2. The scripts will retry up to 5 times, with progressively increasing wait times between attempts.
    
    - Initial Configuration
       1. Update the TEAM_TOKEN in all scripts to your actual team token.
       2. The number_of_runs is set to the maximum possible (5) and can be adjusted as needed.
       
    -  Submission Requirement
       1. Before retrieving discussions for the next round, all runs must submit decisions for every user, 
       even if some users have already been classified as positive (decision = 1).
 

