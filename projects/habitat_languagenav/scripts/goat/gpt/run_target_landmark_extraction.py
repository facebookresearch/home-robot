import os

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_gpt_response(command: str):
    with open(
        "projects/habitat_languagenav/scripts/goat/gpt/target_landmark_prompt.txt"
    ) as f:
        prompt = f.read()

    prompt = prompt + command
    messages = [
        {"role": "user", "content": prompt},
    ]
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages,)[
        "choices"
    ][0]["message"]["content"]

    return response


if __name__ == "__main__":
    # This is the command we want the LLM to parse and generate code for
    commands = [
        [
            "# Can you find my keys? I think I forgot them on my desk in my bedroom. Otherwise, they're probably under a pillow on the sofa.",
            "# Find the curtain. It is located below and to the right of the cardboard box and is fully contained within the frame. The wall is to the left of the curtain and partially overlaps with it.",
        ]
    ]

    for command in commands:
        print(command)
        print("-" * 40)
        print(generate_gpt_response(command))
