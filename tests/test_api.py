import requests

# Replace with your actual API key
API_KEY = "up_k4o4NqbJpz4kzmR9lt9MvGHeQ7xYC"
BASE_URL = "https://api.upstage.ai/v1/solar/chat/completions"  # Assuming the endpoint

# Define your question, completion, and ground truth
question = "Your question here"
completion = "The response you received"
ground_truth = "The correct answer"

# Prepare the messages payload
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant that evaluates the correctness of a response to a question based on the ground truth.",
    },
    {
        "role": "user",
        "content": f"""Please evaluate the correctness of the following response to the question based on the ground truth.

**Question**: {question}

**Response**: {completion}

**Ground truth**: {ground_truth}
You have to return 'yes' if the response is correct, 'no' if it is incorrect. The correct response should have the same meaning as the ground truth; it doesn't need to be exactly the same. Please just return only 'yes' or 'no', don't need to explain.
""",
    },
]

# Set up the headers for authentication and content type
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# Prepare the payload
payload = {
    "model": "solar-pro",
    "messages": messages,
    "stream": True,  # Enable streaming if supported
}

# Make the POST request with streaming
with requests.post(BASE_URL, headers=headers, json=payload, stream=True) as response:
    if response.status_code == 200:
        # Stream and process the response line by line
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                print(decoded_line)
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")