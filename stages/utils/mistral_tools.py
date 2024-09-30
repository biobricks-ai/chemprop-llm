from tqdm import tqdm
import torch, torch.nn.functional as F

def generate_categorical_distribution_logits(model, tokenizer, categories, input_tensor, max_category_length=5):
    """
    Efficiently generate a categorical distribution based on the input context and multi-token categories.
    
    Args:
        model: Pre-trained language model.
        tokenizer: Pre-trained tokenizer associated with the model.
        categories: List of categories (strings) to calculate probabilities for.
        input_tensor: The input tensor (already tokenized) that serves as the context for the model.
        max_category_length: Maximum length of tokenized categories (default is 5).
    
    Returns:
        dict: A dictionary where keys are categories and values are the log-probabilities for each category.
    """

    # Tokenize all categories at once
    category_token_ids = tokenizer(
        categories, 
        add_special_tokens=False, 
        padding="max_length",       # Ensure padding to max_length
        truncation=True,            # Truncate if longer than max_category_length
        max_length=max_category_length,  # Set exact length to 5
        return_tensors="pt"
    ).input_ids.to(model.device)
    
    # strip the sos
    category_token_ids = category_token_ids[:, 1:]
    
    # Prepare input tensor
    batch_size = category_token_ids.size(0)
    current_input_tensor = input_tensor.unsqueeze(0).repeat(batch_size, 1)  # Repeat input for each category

    # Initialize total log-probabilities for each category
    total_log_probs = torch.zeros(batch_size, device=model.device)

    with torch.no_grad():  # Disable gradients for faster inference
        for i in tqdm(range(category_token_ids.size(1))):
            # Get logits for each token in the category (batched across categories)
            outputs = model(current_input_tensor)
            logits = outputs.logits[:, -1, :]  # Logits for the next token
            logprobs = F.log_softmax(logits, dim=-1)
            
            # Get the token IDs for the current position across all categories
            token_ids = category_token_ids[:, i]
            token_logprobs = logprobs.gather(1, token_ids.unsqueeze(-1)).squeeze(-1)

            # Accumulate log-probabilities
            total_log_probs += token_logprobs

            # Append the token ID to the input tensor for the next step
            current_input_tensor = torch.cat((current_input_tensor, token_ids.unsqueeze(-1)), dim=1)

    # Convert to a dictionary
    category_distribution = {category: total_log_probs[i].item() for i, category in enumerate(categories)}

    return category_distribution