import fcntl
import json
import logging
import os
import time
from typing import Any, Dict, List

class DemoChat:
    """
    A class for managing chat logs.

    Attributes:
    -----------
    log_file_path : str
        The path to the chat log file.
    """

    def __init__(self, log_file_path: str) -> None:
        """
        Initializes a Chat instance.

        Parameters:
        -----------
        log_file_path : str
            The path to the chat log file.
        """
        self.log_file_path = log_file_path
        self.last_seen_entry_id = -1

        if not os.path.exists(self.log_file_path):
            with open(self.log_file_path, "w") as f:
                json.dump([], f)

    def input(self, message: str = None, role: str = "system") -> List[Dict]:
        """
        Reads the chat log file and returns its contents.

        Returns:
        --------
        data : list of dict
            The new entries in the chat log file.
        """
        if message:
            self.output(message, role=role)

        # Wait for new entries
        while True:
            time.sleep(2)  # Polling delay
            logging.info("Polling chat log file...")
            with open(self.log_file_path, "r") as f:
                with fcntl.flock(f, fcntl.LOCK_SH):
                    log_data = json.load(f)

            if len(log_data) - 1 > self.last_seen_entry_id:
                new_entries = log_data[self.last_seen_entry_id + 1 :]
                self.last_seen_entry_id = len(log_data) - 1
                return "\n".join([entry["content"] for entry in new_entries])

    def output(self, message: str, role: str = "system") -> None:
        """
        Writes the given role and message to the chat log file.

        Parameters:
        -----------
        role : str
            The role of the message.
        message : str
            The message to write to the chat log file.
        """
        with open(self.log_file_path, "r+") as f:
            with fcntl.flock(f, fcntl.LOCK_EX):
                log_data = json.load(f)
                new_entry = {"role": role, "content": message}
                log_data.append(new_entry)
                f.seek(0)
                json.dump(log_data, f, indent=4)
                f.truncate()
                self.last_seen_entry_id = len(log_data) - 1


if __name__ == "__main__":
    chat = DemoChat("chat_log.json")
    task = chat.input("please type any task you want the robot to do: ")
    chat.output("Plan: for task: " + task[0]["content"])
    execute = chat.input("do you want to execute (replan otherwise)? (y/n): ")
    if "y" in execute[-1]["content"]:
        chat.output("Navigating to instance ")
        chat.output(f"Instance id: {2}")
        chat.output(f"Success: {True}")

    with open("chat_log.json", "r") as f:
        log_data = json.load(f)
        print("Chat log file contents:")
        print(log_data)
