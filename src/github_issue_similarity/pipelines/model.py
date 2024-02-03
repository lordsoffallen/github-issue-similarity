from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer, PreTrainedModel
import torch
from typing import Any
from datasets import Dataset


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_model_and_tokenizer(checkpoint: str) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint)
    model.to(device)

    return model, tokenizer


def cls_pooling(model_output):
    """
    We need to “pool” or average our token embeddings via CLS pooling on our model’s outputs.
    We simply collect the last hidden state for the special [CLS] token.
    """
    return model_output.last_hidden_state[:, 0]


def get_embeddings(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text_list: list[str] | Any
):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)


def compute_embeddings(ds: Dataset, checkpoint: str) -> Dataset:
    model, tokenizer = get_model_and_tokenizer(checkpoint)

    ds = ds.map(
        lambda x: {
            "embeddings": get_embeddings(model, tokenizer, x["text"]).detach().cpu().numpy()[0]
        }
    )

    return ds


