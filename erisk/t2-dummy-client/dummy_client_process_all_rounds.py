import os
import json
import random
from get_discussions import get_discussions, get_list_target_subjects
from submit_run_decisions import submit_decisions, load_user_nicks


class DummyClient:
    def __init__(self, team_token, number_of_runs, discussions_dir="output_discussions", users_dir="dummy_users"):
        """
        Initializes the DummyClient with team token, number of runs, and output directories.

        :param team_token: The unique token for the team.
        :param number_of_runs: The number of runs, max number is 5 runs, and starts from 0.
        :param discussions_dir: Directory to save discussions.
        :param users_dir: Directory to save user files.
        """
        self.team_token = team_token
        self.number_of_runs = number_of_runs
        self.discussions_dir = discussions_dir
        self.users_dir = users_dir

    def process_rounds(self):
        """
        Processes all rounds by submitting decisions for all runs and then fetching discussions.
        """
        while True:
            print(f"Processing discussions for team token: {self.team_token}")

            # Fetch discussions for the current round
            discussions = get_discussions(self.team_token)

            if not discussions:
                print("No more discussions to process. All rounds completed.")
                break

            # Save discussions to a file
            discussion_number = discussions[0]["number"]
            os.makedirs(self.discussions_dir, exist_ok=True)
            output_filename = os.path.join(self.discussions_dir, f"{self.team_token}_discussions_number_{discussion_number}.json")
            with open(output_filename, "w") as json_file:
                json.dump(discussions, json_file, indent=4)
            print(f"Saved discussions to {output_filename}")

            # For the first round, extract and save target subjects
            if discussion_number == 0:
                self.save_target_users(discussions)

            # Load target users for decisions
            target_users = self.load_target_users()

            # Submit decisions for all runs
            for run_id in range(self.number_of_runs):
                decisions = self.create_mock_decisions(target_users)
                print(f"Submitting decisions for run {run_id}...")
                response = submit_decisions(self.team_token, run_id, decisions)

                if response:
                    print(f"Successfully submitted decisions for run {run_id}.")
                else:
                    print(f"Failed to submit decisions for run {run_id}.")
                    return  # Stop processing if a submission fails

    def save_target_users(self, discussions):
        """
        Extracts and saves target users from the discussions to a file.

        :param discussions: The JSON data of discussions.
        """
        target_users_list = get_list_target_subjects(discussions)
        os.makedirs(self.users_dir, exist_ok=True)
        targets_filename = os.path.join(self.users_dir, f"{self.team_token}_target_subjects.txt")
        with open(targets_filename, "w") as targets_file:
            for target_user in target_users_list:
                targets_file.write(f"{target_user}\n")
        print(f"Saved target subjects to {targets_filename}")

    def load_target_users(self):
        """
        Loads target users from the saved file.

        :return: A set of usernames.
        """
        return load_user_nicks(os.path.join(self.users_dir, f"{self.team_token}_target_subjects.txt"))

    def create_mock_decisions(self, target_users):
        """
        Creates mock decisions for the given target users.

        :param target_users: A set of usernames.
        :return: A list of decisions.
        """
        return [
            {
                "nick": nick,
                "decision": random.choice([0, 1]),  # Example decision value
                "score": random.random()  # Example score, just a random number
            }
            for nick in target_users
        ]


# Example usage
if __name__ == "__main__":
    # Initialize the DummyClient with the desired team token and number of runs
    client = DummyClient(team_token="your-token", number_of_runs=5)
    client.process_rounds()
