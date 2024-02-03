from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer, PreTrainedModel
import torch
from typing import Any
from datasets import Dataset
import pandas as pd


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_model_and_tokenizer(checkpoint: str) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint)
    model.to(device)

    return model, tokenizer


def cls_pooling(model_output) -> torch.Tensor:
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
    """ Compute embeddings for the existing GitHub issues """
    model, tokenizer = get_model_and_tokenizer(checkpoint)

    ds = ds.map(
        lambda x: {
            "embeddings": get_embeddings(model, tokenizer, x["text"]).detach().cpu().numpy()[0]
        }
    )

    return ds


def find_similar_issues(
    ds: Dataset, checkpoint: str, query: str, top_k: int = 5
) -> pd.DataFrame:
    """ Apply FAISS index to find similar github issues given a query. """

    ds = ds.add_faiss_index(column='embeddings')

    model, tokenizer = get_model_and_tokenizer(checkpoint)
    embedding = get_embeddings(model, tokenizer, [query]).cpu().detach().numpy()

    scores, samples = ds.get_nearest_examples("embeddings", embedding, k=top_k)

    samples_df = pd.DataFrame.from_dict(samples)
    samples_df["scores"] = scores
    samples_df.sort_values("scores", ascending=False, inplace=True)

    return samples_df


def print_similar_issues(samples_df: pd.DataFrame):
    for _, row in samples_df.iterrows():
        print(f"COMMENT: {row.comments}")
        print(f"SCORE: {row.scores}")
        print(f"TITLE: {row.title}")
        print(f"URL: {row.html_url}")
        print("=" * 50)
        print()
