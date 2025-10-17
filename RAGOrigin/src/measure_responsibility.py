import random
import numpy as np
import torch
from tqdm import tqdm

random.seed(1)

MULTIPLE_PROMPT_1 = 'Below is a query from a user and a relevant context. \
Answer the question given the information in the context. \
\n\nContext: [context] \n\nQuery: [question] \n\nAnswer:'

MULTIPLE_PROMPT_2 = 'Below is a query from a user and a relevant context. \
Answer the question given the information in the context. \
\n\nContext: [context] \n\nQuery:'

def wrap_prompt_1(context, question) -> str:
    return MULTIPLE_PROMPT_1.replace('[context]', context).replace('[question]', question)

def wrap_prompt_2(context) -> str:
    return MULTIPLE_PROMPT_2.replace('[context]', context)


def z_score_normalize(data):
    """Applies z-score normalization to a dataset."""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:  
        return np.zeros_like(data)  
    return (data - mean) / std

def normalize(data, method='min_max'):
    """General normalization function that applies either z-score or min-max."""
    if method == 'z_score':
        return (data - np.mean(data)) / np.std(data)
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def calculate_loss(model, tokenizer, context, response, device):
    """
    Calculates loss for generating 'response' given 'context' using the model.
    
    Args:
        model: Language model 
        tokenizer: Tokenizer associated with the model
        context: Input context
        response: Response to evaluate
        device: Computing device (CPU/GPU)
        
    Returns:
        Loss value as a float
    """
    text = context + ' ' + response
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    # Mask the context tokens to calculate loss only on the response
    context_ids = tokenizer(context, return_tensors="pt")["input_ids"]
    label_ids = input_ids.clone()
    label_ids[:, :context_ids.shape[1]] = -100  # Mask context tokens

    with torch.no_grad():
        outputs = model(input_ids, labels=label_ids)

    return outputs.loss.item()

def calculate_scores(model, tokenizer, contexts, question, RAG_response, retrieval_scores, device):
    """
    Calculate various score types for each context passage.
    
    Args:
        model: Language model
        tokenizer: Tokenizer associated with the model
        contexts: List of context passages
        question: User query
        RAG_response: Response from RAG system
        retrieval_scores: Original retrieval scores for contexts
        device: Computing device
        
    Returns:
        Tuple of (answer_scores, question_scores, retrieval_scores)
    """
    answer_scores = []
    question_scores = []

    for idx, context in enumerate(contexts):
        # Calculate loss for generating the RAG response from context and question
        context_prompt_1 = wrap_prompt_1(context, question)
        answer_loss = calculate_loss(model, tokenizer, context_prompt_1, RAG_response, device)
        answer_scores.append(answer_loss)
        
        # Calculate loss for generating the question from just the context
        context_prompt_2 = wrap_prompt_2(context)
        question_loss = calculate_loss(model, tokenizer, context_prompt_2, question, device)
        question_scores.append(question_loss)

    return answer_scores, question_scores, retrieval_scores


def trace_scores(answer_scores, question_scores, retrieval_scores, normalize_method='z_score', trace_type=0):
    """
    Calculate trace scores using different combinations of score types.
    
    Args:
        answer_scores: List of scores for answer generation
        question_scores: List of scores for question generation
        retrieval_scores: List of retrieval scores
        normalize_method: Method for normalizing scores ('z_score' or 'min_max')
        trace_type: Type of trace score to calculate (0-6)
        
    Returns:
        List of combined scores according to trace_type
    """
    normalize_func = z_score_normalize

    # Note: Negating the answer and question scores since lower loss is better
    answer_scores_norm = normalize_func(-np.array(answer_scores))
    question_scores_norm = normalize_func(-np.array(question_scores))
    retrieval_scores_norm = normalize_func(np.array(retrieval_scores))

    # Different combination methods for trace scores
    if trace_type == 0:
        return [(a + b + c) / 3 for a, b, c in zip(answer_scores_norm, question_scores_norm, retrieval_scores_norm)]
    elif trace_type == 1:
        return answer_scores_norm.tolist()
    elif trace_type == 2:
        return question_scores_norm.tolist()
    elif trace_type == 3:
        return retrieval_scores_norm.tolist()


def measure_responsibility(data, model, tokenizer, device, variant, top_K):
    """
    Measure responsibility scores for all items in the dataset.
    
    Args:
        data: List of data items
        model: Language model
        tokenizer: Tokenizer associated with the model
        device: Computing device
        variant: Which trace score variant to use
        top_K: Number of contexts to consider in each group
        
    Returns:
        Tuple containing various results and metadata
    """
    y_true = []
    trace_scores_dict = {f"variant_{variant}": []}
    scores_iter_result = []
    id_list = []
    scope_sizes = []
    topk_contexts = []
    ids = []
    incorrects = []
    corrects = []
    full_contexts = []
    questions = []
    
    for q_idx in tqdm(range(len(data))):
        item = data[q_idx]
        
        # Determine the scope size based on probe results
        probe_results = item['check_results']
        scope_size = 2
        while sum(probe_results[:scope_size]) != int(scope_size/2):
            scope_size += 2
        scope_size = scope_size * top_K
        if scope_size > len(item['context_texts']):
            scope_size = len(item['context_texts'])
        scope_sizes.append(scope_size)
        
        # Collect metadata
        id_list.append(item['question_id'])
        ids.append(item['question_id'])
        questions.append(item['question'])
        incorrects.append(item['target_answer'])
        corrects.append(item['correct_answer'])
        full_contexts.append(item['context_texts'])

        # Get the relevant contexts within the scope
        context_texts = item['context_texts'][:scope_size]
        topk_contexts.append(context_texts)
        context_labels = item['context_labels'][:scope_size]
        retrieval_scores = item['retrieval_scores'][:scope_size]
        question = item['question']
        RAG_response = item['RAG_response']
        y_true.append(context_labels)
        
        # Calculate various scores
        answer_scores, question_scores, retrieval_scores = calculate_scores(
            model, tokenizer, context_texts, question, RAG_response, retrieval_scores, device
        )
        
        # Calculate trace scores using the specified variant
        iter_trace_scores_dict = {}
        tmp_scores = trace_scores(answer_scores, question_scores, retrieval_scores, 'z_score', variant)
        trace_scores_dict[f"variant_{variant}"].append(tmp_scores)
        iter_trace_scores_dict[f"variant_{variant}"] = tmp_scores
        
        # Store results for this item
        scores_iter_result.append({
            'question_id': item['question_id'],
            'question': question,
            'target_answer': item['target_answer'],
            'RAG_response': RAG_response,
            'scope_size': scope_size,
            'answer_scores': answer_scores,
            'question_scores': question_scores,
            'retrieval_scores': retrieval_scores,
            'trace_scores_dict': iter_trace_scores_dict
        })
        
    return (
        y_true, trace_scores_dict, scores_iter_result, id_list, 
        scope_sizes, topk_contexts, ids, questions, incorrects, corrects, full_contexts
    )
