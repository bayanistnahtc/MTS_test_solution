from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
import torch

# Load the dataset
dataset = load_dataset("databricks/databricks-dolly-15k")

model_id = "HuggingFaceH4/zephyr-7b-beta"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16
)


MAX_CONTEXT_LENGTH = 1024
MAX_INSTRUCTION_LENGTH = 512


def truncate_examples(examples):
    truncated_contexts = []
    truncated_instructions = []

    for context, instruction in zip(examples['context'], examples['instruction']):
        if context:
            context = context[:MAX_CONTEXT_LENGTH]

        instruction = instruction[:MAX_INSTRUCTION_LENGTH]

        truncated_contexts.append(context)
        truncated_instructions.append(instruction)

    return {
        'context': truncated_contexts,
        'instruction': truncated_instructions
    }


# Apply truncation to the entire dataset
dataset = dataset["train"].map(
    truncate_examples,
    batched=True,
    batch_size=1000
)

#dataset  = dataset.select(range(1000))

# Create text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=10,
    # temperature=0.1
)


classification_prompt = """You will be provided with a query.
Your task is to determine whether answering this query requires retrieving additional external information beyond general world knowledge.

Respond with 'retrieval_required' if the query:

- Requires up-to-date information (e.g., current events, recent data, or facts that may change over time).
- Requires specific domain knowledge or specialized information not commonly known.
- Involves factual verification or sourcing.


Respond with 'no_retrieval_required' if the query can be answered using general world knowledge, common sense, or information already contained in the context.

Consider the following examples:
- "Who is the current president of the USA?" → retrieval_required (requires up-to-date information).
- "Who wrote 'Romeo and Juliet'?" → no_retrieval_required (general knowledge).
- "What are the latest COVID-19 statistics?" → retrieval_required (requires up-to-date information).
- "What is the square root of 64?" → no_retrieval_required (general knowledge).

Important Instructions:
- Respond exclusively with either 'retrieval_required' or 'no_retrieval_required'.
- Do not provide any explanations, additional text, or examples.
- Do not generate any text beyond the required response.

Query: {context}{instruction}
Response:"""


def label_retrieval_requirement(batch):
    answers = []
    labels = []
    for instruction, context in zip(batch["instruction"], batch["context"]):
        prompt = classification_prompt.format(
            instruction=instruction,
            context=f"\n\n{context}" if context else ""
        )

        response = pipe(prompt, do_sample=False, return_full_text=False)
        answer = response[0]['generated_text'].strip().lower()
        answers.append(answer)

        if "no_retrieval_required" in answer:
            labels.append(0)
        elif "retrieval_required" in answer:
            labels.append(1)
        else:
            labels.append(-1)

    return {"retrieval_label": labels, "answers": answers}


labeled_dataset = dataset.map(
    label_retrieval_requirement,
    batched=True,
    batch_size=128
)

# Filter out ambiguous cases
filtered_dataset = labeled_dataset.filter(
    lambda x: x["retrieval_label"] != -1
)

filtered_dataset.save_to_disk("dolly15k_retrieval_labeled")
