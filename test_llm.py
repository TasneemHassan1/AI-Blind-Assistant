from ollama_llm import OllamaLLM

scene = {
    1: {
        "name": "cup",
        "position": "left",
        "support_object": "table",
    },
    2: {
        "name": "laptop",
        "position": "ahead",
    },
}

memory = {}

llm = OllamaLLM()

print(
    llm.generate_response(
        "Where is the cup?",
        scene,
        memory,
    )
)