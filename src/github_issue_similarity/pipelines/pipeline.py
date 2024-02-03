from kedro.pipeline import Pipeline, pipeline, node
from .data import fetch_issues, clean, preprocess
from .model import compute_embeddings, find_similar_issues, print_similar_issues


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
            name='compute_embeddings'
        ),
        node(
            find_similar_issues,
            inputs=[
                'embeddings',
                'params:model_checkpoint',
                'params:query',
            ],
            outputs='similar_issues',
            name='find_similar_issues'
        ),
        node(
            print_similar_issues,
            inputs='similar_issues',
            outputs=None,
            name='print_similar_issues'
        ),
    ])
