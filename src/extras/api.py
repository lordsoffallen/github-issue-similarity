from kedro_datasets.api.api_dataset import APIDataset
from kedro.io.core import DatasetError
from typing import Any, Callable
from tqdm.auto import tqdm
from requests.auth import AuthBase

import requests
import math
import time


class GitHubIssueAPIDataset(APIDataset):
    def __init__(  # noqa: PLR0913
        self,
        *,
        url: str,
        method: str = "GET",
        owner: str = None,
        repo: str = None,
        load_args: dict[str, Any] = None,
        save_args: dict[str, Any] = None,
        credentials: tuple[str, str] | list[str] | AuthBase = None,
        metadata: dict[str, Any] = None,
    ) -> None:

        super().__init__(
            url=url,
            method=method,
            credentials=credentials,
            save_args=save_args,
            load_args=load_args,
            metadata=metadata
        )

        self._owner = owner
        self._repo = repo

    def _load(self) -> list[requests.Response]:
        if self._request_args["method"] == "GET":
            self._request_args.pop("method")

            batch = []
            all_issues = []
            per_page = self._request_args.pop("num_issues_per_page")  # Number of issues to return per page
            num_issues = self._request_args.pop("num_issues")
            num_pages = math.ceil(num_issues / per_page)
            base_url = self._request_args.pop("url")
            rate_limit = self._request_args.pop('rate_limit', 5_000)

            for i, page in enumerate(tqdm(range(num_pages))):
                # Query with state=all to get both open and closed issues
                query = f"issues?page={page}&per_page={per_page}&state=all"
                issues = requests.get(f"{base_url}/{self._owner}/{self._repo}/{query}", **self._request_args)
                batch.extend(issues.json())

                if i + 1 >= rate_limit:
                    all_issues.extend(batch)
                    batch = []  # Flush batch for next time period
                    print(f"Reached GitHub rate limit. Sleeping for one hour ...")
                    time.sleep(60 * 60 + 1)

            all_issues.extend(batch)

            return all_issues

        raise DatasetError("Only GET method is supported for load")


class GitHubCommentAPIDataset(APIDataset):
    def __init__(  # noqa: PLR0913
        self,
        *,
        url: str,
        method: str = "GET",
        owner: str = None,
        repo: str = None,
        load_args: dict[str, Any] = None,
        save_args: dict[str, Any] = None,
        credentials: tuple[str, str] | list[str] | AuthBase = None,
        metadata: dict[str, Any] = None,
    ) -> None:

        super().__init__(
            url=url,
            method=method,
            credentials=credentials,
            save_args=save_args,
            load_args=load_args,
            metadata=metadata
        )

        self._owner = owner
        self._repo = repo

    def _load(self) -> Callable:
        if self._request_args["method"] == "GET":
            self._request_args.pop("method")

            base_url = self._request_args.pop("url")

            def comment_getter(issue_number):
                url = f"{base_url}/{self._owner}/{self._repo}/issues/{issue_number}/comments"
                response = requests.get(url, **self._request_args)
                return [r["body"] for r in response.json()]

            return comment_getter

        raise DatasetError("Only GET method is supported for load")

