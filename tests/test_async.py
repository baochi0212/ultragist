# tests/test_api_async.py (1-140)
import asyncio
import aiohttp
import json
from aiohttp import ClientSession, ClientResponseError
import sys

# Replace with your actual API key
API_KEY = "up_k4o4NqbJpz4kzmR9lt9MvGHeQ7xYC"
BASE_URL = "https://api.upstage.ai/v1/solar/chat/completions"  # Assuming the endpoint

# Define your question, completion, and ground truth
question = "What is the capital of France?"
completion = "The capital of France is Paris."
ground_truth = "Paris is the capital city of France."

# Number of concurrent requests you want to make
NUM_REQUESTS = 10  # Adjust as needed

# Semaphore to limit the number of concurrent requests (to respect API rate limits)
SEM_LIMIT = 10  # Adjust based on API's rate limits

async def create_messages(question: str, completion: str, ground_truth: str):
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

async def make_request(session: ClientSession, request_id: int, semaphore: asyncio.Semaphore):
    payload = {
        "model": "solar-pro",
        "messages": await create_messages(question, completion, ground_truth),
        "stream": False,  # Enable streaming if supported
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    async with semaphore:
        try:
            async with session.post(BASE_URL, headers=headers, json=payload) as response:
                if response.status != 200:
                    text = await response.text()
                    print(f"Request {request_id} failed with status code {response.status}: {text}")
                    return

                print(f"Request {request_id} succeeded:")
                # Assuming the response is in JSON Lines (each line is a JSON object)
                async for line in response.content:
                    if line:
                        decoded_line = line.decode('utf-8').strip()
                        if decoded_line:
                            # Depending on the API's streaming format, you might need to parse accordingly
                            try:
                                data = json.loads(decoded_line)
                                print(f"Request {request_id} response: {json.dumps(data)}")
                            except json.JSONDecodeError:
                                # If the streaming format is not JSON, simply print the raw line
                                print(f"Request {request_id} response: {decoded_line}")
        except ClientResponseError as e:
            print(f"Request {request_id} encountered a client error: {e}")
        except asyncio.TimeoutError:
            print(f"Request {request_id} timed out.")
        except Exception as e:
            print(f"Request {request_id} encountered an unexpected exception: {e}")

async def main():
    semaphore = asyncio.Semaphore(SEM_LIMIT)  # Limit concurrent requests
    timeout = aiohttp.ClientTimeout(total=60)  # Adjust timeout as needed

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            asyncio.create_task(make_request(session, request_id, semaphore))
            for request_id in range(1, NUM_REQUESTS + 1)
        ]

        # Optionally, you can gather tasks and handle exceptions collectively
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for request_id, result in enumerate(results, start=1):
            if isinstance(result, Exception):
                print(f"Request {request_id} generated an exception: {result}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)