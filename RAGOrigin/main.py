import argparse
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.determine_threshold import evaluate_results
from src.measure_responsibility import measure_responsibility
from src.narrow_scope import narrow_scope
from src.OpenAI_API import OpenAIGenerator

def split_y_pred(y_pred, scope_sizes):
    """Split the flat prediction array into chunks based on scope sizes."""
    result = []
    start_index = 0
    
    for size in scope_sizes:
        end_index = start_index + size
        if end_index > len(y_pred):
            end_index = len(y_pred)
        result.append(y_pred[start_index:end_index])
        start_index = end_index
        
        if start_index >= len(y_pred):
            break
            
    return result


def generate_config(model_name, batch_size=10, temperature=0.1, top_p=0.1):
    """Generate configuration dictionary for the LLM API."""
    return {
        "generator_model": model_name,
        "generator_batch_size": batch_size,
        "generation_params": {
            "temperature": temperature,
            "top_p": top_p,
        },
        "openai_setting": {
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "base_url": os.environ.get("OPENAI_API_URL")
        },
    }

def setup_paths(args):
    """Set up directory paths for results and feedback files."""
    result_dir = f'{args.result_root_dir}/{args.dataset}/{args.attack_method}/{args.trace_method}/{args.attack_retriever}_{args.attack_LLM}_{args.top_K}_{args.attack_M}_{args.test_version}'
    if not os.path.exists(result_dir): 
        os.makedirs(result_dir)

    feedback_file_path = f"{args.feedback_root_dir}/{args.dataset}/{args.attack_method}/k{args.top_K}_m{args.attack_M}_{args.attack_retriever}_{args.attack_LLM}.json"
    feedback_scope_file_path = f"{args.feedback_scope_dir}/{args.dataset}/{args.attack_method}/k{args.top_K}_m{args.attack_M}_{args.attack_retriever}_{args.attack_LLM}.json"
    
    feedback_scope_file_dir = os.path.dirname(feedback_scope_file_path)
    os.makedirs(feedback_scope_file_dir, exist_ok=True)
    
    return result_dir, feedback_file_path, feedback_scope_file_path


def setup_models(args):
    """Initialize and set up all required models."""
    # Set up OpenAI API generators
    config = generate_config(args.attack_LLM)
    config_judge_llm = generate_config(args.judge_LLM)
    generator = OpenAIGenerator(config)
    judge_llm = OpenAIGenerator(config_judge_llm)
    
    # Load local proxy model for responsibility measurement
    model_name_or_path = f'{args.proxy_model}'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    device = f"cuda:{args.cuda_device}"
    model.to(device)
    model.eval()
    
    return generator, judge_llm, model, tokenizer, device


def load_narrowed_data(feedback_file_path, feedback_scope_file_path, generator, judge_llm, top_K):
    """Load or create narrowed scope data."""
    if not os.path.exists(feedback_scope_file_path):
        data = narrow_scope(feedback_file_path, feedback_scope_file_path, generator, judge_llm, top_K)
    else:
        with open(feedback_scope_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    return data


def main():
    parser = argparse.ArgumentParser(description='Experiment')
    parser.add_argument('--dataset', type=str, default="NQ", help='Dataset name')
    parser.add_argument('--attack_retriever', type=str, default="e5", help='Attacked retriever model')
    parser.add_argument('--attack_LLM', type=str, default="gpt-4o-mini", help='Attacked LLM')
    parser.add_argument('--judge_LLM', type=str, default="gpt-4o-mini", help='LLM used for judging consistency')
    parser.add_argument('--attack_method', type=str, default="PRAGB", help='Attack method name')
    parser.add_argument('--attack_M', type=int, default=5, help='Number of poisoned documents')
    parser.add_argument('--top_K', type=int, default=5, help='Number of documents to retrieve')
    parser.add_argument('--trace_method', type=str, default="RAGOrigin", help='Tracing method')
    parser.add_argument('--proxy_model', type=str, default="meta-llama/Llama-3.1-8B", help='Model for responsibility measurement')
    parser.add_argument('--variant', type=int, default=0, help='Variant of trace score calculation')  
    parser.add_argument('--normalize_method', type=str, default="z_score_normalize", help='Method for normalizing scores')
    parser.add_argument('--feedback_root_dir', type=str, default="attack_feedback", help='Root directory for feedback files')
    parser.add_argument('--feedback_scope_dir', type=str, default="attack_feedback_scope", help='Directory for narrowed scope files')
    parser.add_argument('--result_root_dir', type=str, default="result", help='Root directory for results')
    parser.add_argument('--test_version', type=str, default="v1", help='Test version identifier')
    parser.add_argument('--cuda_device', type=int, default=1, help='CUDA device index')  
    args = parser.parse_args()
    
    # Set up paths and models
    result_dir, feedback_file_path, feedback_scope_file_path = setup_paths(args)
    generator, judge_llm, model, tokenizer, device = setup_models(args)
    
    # Load or create narrowed data
    data = load_narrowed_data(feedback_file_path, feedback_scope_file_path, generator, judge_llm, args.top_K)
    
    # Measure responsibility scores
    (y_true, trace_scores_dict, scores_iter_result, id_list, scope_sizes, 
     topk_contexts, ids, questions, incorrects, corrects, full_contexts) = measure_responsibility(
        data, model, tokenizer, device, args.variant, args.top_K
    )
    
    # Evaluate results using dynamic thresholding
    tn_list, fp_list, fn_list, tp_list, fpr, fnr, accuracy, y_pred = evaluate_results(
        y_true, trace_scores_dict, args.variant
    )
    
    # Split predictions back into per-query groups
    split_result = split_y_pred(y_pred, scope_sizes)
    
    dynamic_threshold_result = {
        'DACC': accuracy,
        'FPR': fpr, 
        'FNR': fnr,
        'id_list': id_list,
        'TN_list': [int(e) for e in tn_list], 
        'FP_list': [int(e) for e in fp_list],
        'FN_list': [int(e) for e in fn_list], 
        'TP_list': [int(e) for e in tp_list],
    }
    
    # Extract top-k clean contexts based on predictions
    topk = []
    for idx, i in enumerate(split_result):
        tmp = []
        for idx2, i2 in enumerate(split_result[idx]):
            if split_result[idx][idx2] == False:  # False means clean context
                tmp.append(full_contexts[idx][idx2])
            if len(tmp) == args.top_K:
                break
        topk.append(tmp)
    print(f"Top-k length: {len(topk)}")
    
    context_result = []
    for idx, i in enumerate(ids):
        context_result.append({
            'question_id': ids[idx],
            'question': questions[idx],
            'correct_answer': corrects[idx],
            'incorrect_answer': incorrects[idx],
            'clean_topk': topk[idx],
            'contexts_labels': split_result[idx],
            "contexts": full_contexts[idx],
        })
    
    print(f"[Dynamic Threshold] FPR: {fpr}, FNR: {fnr}, DACC: {accuracy}")
    
    metric_result = {
        'dynamic_threshold_result': dynamic_threshold_result,
    }

    # Save results to files
    with open(f'{result_dir}/scores_iter_result.json', 'w') as f:
        json.dump(scores_iter_result, f, indent=4)

    with open(f'{result_dir}/metric_result.json', 'w') as f:
        json.dump(metric_result, f, indent=4)
        
    with open(f'{result_dir}/context_result.json', 'w') as f:
        json.dump(context_result, f, indent=4)

if __name__ == '__main__':
    main()
