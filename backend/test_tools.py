import httpx
import json


payload = {
    "model": "qwen2.5-coder:7b",
    "messages": [
        {
            "role": "user",
            "content": "Search the uploaded project for the word encryption."
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
                        "project_id": {
                            "type": "string"
                        },
                        "query": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "project_id",
                        "query"
                    ]
                }
            }
        }
    ],
    "stream": False,
}


response = httpx.post(
    "http://localhost:11434/api/chat",
    json=payload,
    timeout=None,
)

print("STATUS:", response.status_code)
print()
print(json.dumps(response.json(), indent=2))