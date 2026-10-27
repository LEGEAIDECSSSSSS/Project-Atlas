
import json

import httpx


def main():
    payload = {
        "model": "qwen2.5-coder:7b",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Search the uploaded project for the word encryption."
                ),
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search_project",
                    "description": "Search an uploaded project for text.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "project_id": {"type": "string"},
                            "query": {"type": "string"},
                        },
                        "required": ["project_id", "query"],
                    },
                },
            }
        ],
        "stream": False,
    }

    try:
        response = httpx.post(
            "http://localhost:11434/api/chat",
            json=payload,
            timeout=60,
        )
        print("STATUS:", response.status_code)
        print()

        try:
            print(json.dumps(response.json(), indent=2))
        except ValueError:
            print(response.text)

    except httpx.HTTPError as error:
        print(f"Could not reach Ollama: {error}")


if __name__ == "__main__":
    main()
