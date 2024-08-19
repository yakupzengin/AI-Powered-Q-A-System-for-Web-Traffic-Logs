from transformers import T5ForConditionalGeneration, T5Tokenizer

def load_t5_model(model_path="t5-base"):
    """
    Loads the T5 model and tokenizer.

    :param model_path: Path to the pre-trained model
    :return: T5 model and tokenizer
    """
    print("Loading T5 model...")
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    print("T5 model loaded successfully.")
    return model, tokenizer
