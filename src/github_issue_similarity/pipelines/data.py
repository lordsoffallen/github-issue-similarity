from requests import Response
import pandas as pd
from datasets import Dataset
from typing import Callable


def fetch_issues(issues: list[Response]) -> pd.DataFrame:
    return pd.DataFrame.from_records(issues)


def clean(issues: pd.DataFrame, comments_getter: Callable) -> Dataset:
    ds = Dataset.from_pandas(issues, split='train')

    # Remove pull requests from the issue data
    ds = ds.map(lambda x: {"is_pull_request": False if x["pull_request"] is None else True})
    ds = ds.filter(lambda x: x["is_pull_request"] is False)
    ds = ds.map(lambda x: {"comments": comments_getter(x["number"])})
    ds = ds.filter(lambda x: len(x["comments"]) > 0)

    return ds


def preprocess(issues: Dataset) -> Dataset:
    # Drop columns
    columns = issues.column_names
    columns_to_keep = ["title", "body", "html_url", "comments"]
    columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
    ds = issues.remove_columns(columns_to_remove)

    # Explode comments
    ds.set_format('pandas')
    df = ds[:]
    comments_df = df.explode("comments", ignore_index=True)
    ds = Dataset.from_pandas(comments_df)

    ds = ds.map(lambda x: {"comment_length": len(x["comments"].split())})
    ds = ds.filter(lambda x: x["comment_length"] > 15)

    print(ds)

    # concatenate
    def concatenate_text(examples):
        return {
            "text":
                (examples["title"] or "") + " \n "
                + (examples["body"] or "") + " \n "
                + (examples["comments"] or "")
        }

    ds = ds.map(concatenate_text)

    return ds
