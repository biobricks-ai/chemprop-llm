from tqdm import tqdm
from transformers import AutoTokenizer
import torch, torch.nn.functional as F

def generate_categorical_distribution_logits(model, tokenizer, categories, input_tensor):
    """
    Generate a categorical distribution based on the input text for multi-token categories.
    
    Args:
        model: Pre-trained language model.
        tokenizer: Pre-trained tokenizer associated with the model.
        categories: List of categories (strings) that we want to calculate probabilities for.
        input_tensor: The input tensor (already tokenized) that serves as the context for the model.
    
    Returns:
        dict: A dictionary where the keys are the categories and the values are the log-probabilities for each category.
    """
    
    category_distribution = {}

    for category in tqdm(categories):
        category_tokens = tokenizer.encode(category, add_special_tokens=False)
        category_token_ids = torch.tensor(category_tokens).unsqueeze(0).to(model.device)
        
        total_log_prob = 0.0  # Initialize total log-probability for this category
        current_input_tensor = input_tensor.clone().unsqueeze(0)  # Clone the input to avoid modifying the original

        # Generate logits for each token in the category
        token_id = next(iter(category_token_ids[0]))
        for token_id in category_token_ids[0]:
            outputs = model(current_input_tensor)  # Get logits for the current input
            logits = outputs.logits[:, -1, :]  # Logits for the next token
            
            # Convert logits to probabilities using softmax and take the log
            token_prob = F.softmax(logits, dim=-1)
            token_log_prob = torch.log(token_prob[0, token_id])
            
            total_log_prob += token_log_prob  # Sum the log-probabilities
            
            # Append the token to the input for the next token generation
            current_input_tensor = torch.cat((current_input_tensor, token_id.unsqueeze(0).unsqueeze(0)), dim=1)

        # Store the total log-probability for the category
        category_distribution[category] = total_log_prob
    
    return category_distribution