import requests

from app.config import OLLAMA_GENERATE_URL, LLM_MODEL


def generate_answer(prompt):
    response = requests.post(
        OLLAMA_GENERATE_URL,
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=600
    )

    response.raise_for_status()

    data = response.json()

    return data["response"]


if __name__ == "__main__":

    prompt = "Explain RAG in 3 simple sentences."

    answer = generate_answer(prompt)

    print("\nOLLAMA RESPONSE")
    print("=" * 70)
    print(answer)