from kedro.pipeline import Pipeline, pipeline, node
from .data import fetch_issues, clean, preprocess
from .model import compute_embeddings


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(fetch_issues, inputs='issues#api', outputs='issues#json', name='fetch_issues'),
        node(
            clean,
            inputs=dict(issues='issues#json', comments_getter='comments#api'),
            outputs='issues_with_comments',
            name='clean'
        ),
        node(
            preprocess, inputs='issues_with_comments', outputs='preprocessed', name='preprocess'
        ),
        node(
            compute_embeddings,
            inputs=['preprocessed', 'params:model_checkpoint'],
            outputs='embeddings',
            name='embeddings')
    ])
