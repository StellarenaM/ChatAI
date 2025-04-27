import transformers

chat_tokenizer_dir = "./deepseek_v3_tokenizer/"

tokenizer = transformers.AutoTokenizer.from_pretrained(chat_tokenizer_dir,
                                                       trust_remote_code=True)


def get_token_length(messages):
    tokens = tokenizer.apply_chat_template(messages,
                                           tokenize=True,
                                           add_generation_prompt=False)
    return len(tokens)
