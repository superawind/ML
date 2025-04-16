import re

def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    # pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    # completion_contents = [completion[0]["content"] for completion in completions]
    # matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    # return [1.0 if match else 0.0 for match in matches]
    return [1.0 if len(elem)>50 else 0 for elem in completions]

def format_reward2(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    # pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    # completion_contents = [completion[0]["content"] for completion in completions]
    # matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    # return [1.0 if match else 0.0 for match in matches]
    return [2.1 if len(elem)>50 else 1.3 for elem in completions]