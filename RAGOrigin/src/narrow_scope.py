import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

MULTIPLE_PROMPT = 'Below is a query from a user and some relevant contexts. \
Answer the question given the information in those contexts. Your answer should be short and concise. \
If you cannot find the answer to the question, just say "I don\'t know". \
\n\nContexts: [context] \n\nQuery: [question] \n\nAnswer:'

def wrap_prompt(question, context) -> str:
    assert type(context) == list
    context_str = "\n".join(context)
    input_prompt = MULTIPLE_PROMPT.replace('[question]', question).replace('[context]', context_str)
    return input_prompt

def construct_input_list(question, context):
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": wrap_prompt(question, context)}
    ]
        
def check(reference_answer, RAG_response):
    """Creates a prompt to check if two answers are consistent."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": (
            f'Given answer1 and answer2, answer1 is "{reference_answer}" and answer2 is "{RAG_response}". '
            'If answer2 is exactly "I don\'t know", no matter what answer1 is, you must only just return "[Label: False]". '
            'If answer1 is **the same as or consistant with** answer2, return "[Label: True]". '
            'If answer1 is not consistant with answer2, return "[Label: False]".\n'
            'Just give me the answer.'
        )}
    ]

def process_item(item, generator, judge_LLM, top_K):
    """
    Process a single data item by generating answers from different context chunks
    and checking consistency with the original RAG response.
    
    Args:
        item: Dictionary containing question, context, and original response
        generator: LLM for generating answers
        judge_LLM: LLM for judging answer consistency
        top_K: Number of context passages to use in each group
        
    Returns:
        Item with added check_answers and check_results fields
    """
    item['check_answers'] = []
    item['check_results'] = []
    item['check_answers'].append(item['RAG_response'])
    item['check_results'].append(1)
    
    question = item['question']
    context_texts = item['context_texts']
    
    for i in range(top_K, len(context_texts), top_K):
        # Stop when we have equal number of consistent and inconsistent results
        if item['check_results'] and item['check_results'].count(0) == item['check_results'].count(1):
            break
        print("judging")
        grouped_contexts = context_texts[i:i+top_K]
        input_list = construct_input_list(question, grouped_contexts)
        check_answers = generator.generate([input_list])
        check_input_list = [check(item['RAG_response'], check_answers[0])]
        check_results = judge_LLM.generate(check_input_list)
        match = re.search(r'\[Label: (True|False)\]', check_results[0])
        l = 0
        if match:
            label = match.group(1)
            l = 1 if label == 'True' else 0
        item['check_answers'].append(check_answers[0])
        item['check_results'].append(l)
    return item

def narrow_scope(input_file, feedback_scope_file_path, generator, judge_LLM, top_K):
    """
    Process all items in the input file using parallel execution.
    
    Args:
        input_file: Path to input JSON file
        feedback_scope_file_path: Path to save output JSON
        generator: LLM for generating answers
        judge_LLM: LLM for judging answer consistency
        top_K: Number of context passages to use in each group
        
    Returns:
        List of processed data items
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process items in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_item, item, generator, judge_LLM, top_K) for item in data]
        processed_data = []
        for future in tqdm(as_completed(futures), total=len(futures), unit="item"):
            processed_data.append(future.result())
            
    with open(feedback_scope_file_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

    return processed_data
