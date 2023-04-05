#!/usr/bin/env python3
import os
import openai
import pinecone
import time
import sys
from collections import deque
from typing import Dict, List
from dotenv import load_dotenv
import os
import streamlit as st

# Set Variables
load_dotenv()

# Set API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
assert OPENAI_API_KEY, "OPENAI_API_KEY environment variable is missing from .env from .env"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
assert PINECONE_API_KEY, "PINECONE_API_KEY environment variable is missing from .env"

PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east1-gcp")
assert PINECONE_ENVIRONMENT, "PINECONE_ENVIRONMENT environment variable is missing from .env"

# Table config
YOUR_TABLE_NAME = os.getenv("TABLE_NAME", "")
assert YOUR_TABLE_NAME, "TABLE_NAME environment variable is missing from .env"

# Project config
OBJECTIVE = sys.argv[1] if len(sys.argv) > 1 else os.getenv("OBJECTIVE", "")
assert OBJECTIVE, "OBJECTIVE environment variable is missing from .env"

YOUR_FIRST_TASK = os.getenv("FIRST_TASK", "")
assert YOUR_FIRST_TASK, "FIRST_TASK environment variable is missing from .env"


# Configure OpenAI and Pinecone
openai.api_key = OPENAI_API_KEY
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# Create Pinecone index
table_name = YOUR_TABLE_NAME
dimension = 1536
metric = "cosine"
pod_type = "p1"
if table_name not in pinecone.list_indexes():
    pinecone.create_index(table_name, dimension=dimension,
                          metric=metric, pod_type=pod_type)

# Connect to the index
index = pinecone.Index(table_name)

# Task list
task_list = deque([])


def add_task(task: Dict):
    task_list.append(task)


def get_ada_embedding(text):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model="text-embedding-ada-002")["data"][0]["embedding"]


# Dropdown to pick between two gpt models:
pick_model = st.sidebar.selectbox(
    'Pick a model',
    ('gpt-3.5-turbo', 'gpt-4')
)

# Dropdown to pick between game systems:
pick_system = st.sidebar.selectbox(
    'Pick a system',
    ('Call of Cthulhu', 'Dungeons & Dragons', 'Pathfinder')
)
# Set the objective based on the system:
OBJECTIVE = f"Create a character for {pick_system}"


# Optional user input for the first task:
character_preferences = st.sidebar.text_input(
    'Enter your first task',
    'Describe your character'
)


YOUR_FIRST_TASK = f"Create a character backstory based on the description: {character_preferences}"


# The main function that calls GPT
def openai_call(prompt: str, temperature: float = 0.5, max_tokens: int = 100):
    # Call GPT-4 chat model
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=pick_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
        stop=None,
    )
    return response.choices[0].message.content.strip()


def task_creation_agent(objective: str, result: Dict, task_description: str, task_list: List[str], gpt_version: str = 'gpt-3'):
    prompt = f"You are an task creation AI that uses the result of an execution agent to create new tasks with the following objective: {objective}, The last completed task has the result: {result}. This result was based on this task description: {task_description}. These are incomplete tasks: {', '.join(task_list)}. Based on the result, create new tasks to be completed by the AI system that do not overlap with incomplete tasks. Return the tasks as an array."
    response = openai_call(prompt)
    new_tasks = response.split('\n')
    return [{"task_name": task_name} for task_name in new_tasks]


def prioritization_agent(this_task_id: int, gpt_version: str = 'gpt-3'):
    global task_list
    task_names = [t["task_name"] for t in task_list]
    next_task_id = int(this_task_id)+1
    prompt = f"""You are an task prioritization AI tasked with cleaning the formatting of and reprioritizing the following tasks: {task_names}. Consider the ultimate objective of your team:{OBJECTIVE}. Do not remove any tasks. Return the result as a numbered list, like:
    #. First task
    #. Second task
    Start the task list with number {next_task_id}."""
    response = openai_call(prompt)
    new_tasks = response.split('\n')
    task_list = deque()
    for task_string in new_tasks:
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = task_parts[0].strip()
            task_name = task_parts[1].strip()
            task_list.append({"task_id": task_id, "task_name": task_name})


def execution_agent(objective: str, task: str, gpt_version: str = 'gpt-3') -> str:
    # context = context_agent(index="quickstart", query="my_search_query", n=5)
    context = context_agent(index=YOUR_TABLE_NAME, query=objective, n=5)
    # print("\n*******RELEVANT CONTEXT******\n")
    # print(context)
    prompt = f"You are an AI who performs one task based on the following objective: {objective}.\nTake into account these previously completed tasks: {context}\nYour task: {task}\nResponse:"
    return openai_call(prompt, 0.7, 2000)


def context_agent(query: str, index: str, n: int):
    query_embedding = get_ada_embedding(query)
    index = pinecone.Index(index_name=index)
    results = index.query(query_embedding, top_k=n,
                          include_metadata=True)
    # print("***** RESULTS *****")
    # print(results)
    sorted_results = sorted(
        results.matches, key=lambda x: x.score, reverse=True)
    return [(str(item.metadata['task'])) for item in sorted_results]


# Wrap the main loop in a function
st.sidebar.markdown("## Task List")

# Define task_id_counter outside the run_main_loop function
task_id_counter = 1


def run_main_loop():
    # Add the first task as a text input for the user
    first_task = {
        "task_id": 1,
        "task_name": st.sidebar.text_input("Add the first task", value="")
    }

    add_task(first_task)
    global task_id_counter

    while task_list:

        # Display the task list in the sidebar

        for t in task_list:
            st.sidebar.write(f"{t['task_id']}: {t['task_name']}")

        # Step 1: Pull the first task
        task = task_list.popleft()
        st.write(f"## Task {task['task_id']}: {task['task_name']}")

        # Send to execution function to complete the task based on the context
        result = execution_agent(OBJECTIVE, task["task_name"])
        this_task_id = int(task["task_id"])
        st.write("### Task Result:")
        st.write(result)

        # Step 2: Enrich result and store in Pinecone
        enriched_result = {'data': result}
        result_id = f"result_{task['task_id']}"
        vector = enriched_result['data']
        index.upsert([(result_id, get_ada_embedding(vector), {
                     "task": task['task_name'], "result":result})])

        # Step 3: Create new tasks and reprioritize task list
        new_tasks = task_creation_agent(OBJECTIVE, enriched_result, task["task_name"], [
            t["task_name"] for t in task_list])

        for new_task in new_tasks:
            task_id_counter += 1
            new_task.update({"task_id": task_id_counter})
            add_task(new_task)
        prioritization_agent(this_task_id)

        # Clear the sidebar to update it with the new task list
        st.sidebar.empty()


# Add a button to trigger the main loop
if st.button("Create a Character"):
    run_main_loop()
