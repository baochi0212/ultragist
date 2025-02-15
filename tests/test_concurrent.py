# tests/test_api_concurrent.py (1-98)
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Replace with your actual API key
API_KEY = "up_k4o4NqbJpz4kzmR9lt9MvGHeQ7xYC"
BASE_URL = "https://api.upstage.ai/v1/solar/chat/completions"  # Assuming the endpoint

# Define your question, completion, and ground truth
question = "What is the capital of France?"
completion = "The capital of France is Paris."
ground_truth = "Paris is the capital city of France."

# Prepare the messages payload template
def create_messages(question, completion, ground_truth):
    return [
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

# Function to make a single API request
def make_request(request_id):
    payload = {
        "model": "solar-pro",
        "messages": create_messages(question, completion, ground_truth),
        "stream": True,  # Enable streaming if supported
    }

    try:
        with requests.post(BASE_URL, headers=headers, json=payload, stream=True) as response:
            if response.status_code == 200:
                print(f"Request {request_id} succeeded:")
                # Stream and process the response line by line
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        print(f"Request {request_id} response: {decoded_line}")
            else:
                print(f"Request {request_id} failed with status code {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Request {request_id} encountered an exception: {e}")

# Number of concurrent requests you want to make
NUM_REQUESTS = 10

def main():
    with ThreadPoolExecutor(max_workers=NUM_REQUESTS) as executor:
        # Submit all requests to the executor
        futures = {executor.submit(make_request, i): i for i in range(1, NUM_REQUESTS + 1)}
        
        # Optionally, you can process the results as they complete
        for future in as_completed(futures):
            request_id = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"Request {request_id} generated an exception: {exc}")

if __name__ == "__main__":
    main()