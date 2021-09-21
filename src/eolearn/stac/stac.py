"""This module contains STAC related classes and functions."""

__author__ = "Damián Silvani"
__copyright__ = "Dymaxion Labs"
__license__ = "MIT"

import concurrent.futures
import json
import logging
import os
import shutil
from functools import partial
from typing import List

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from .utils import flatten

LOGGER = logging.getLogger(__name__)


class STACSearchRequest:
    def __init__(
        self,
        bbox=None,
        time_interval=None,
        data_folder=None,
        assets=None,
        limit=10,
        *,
        catalog_url,
        collection_name,
    ):
        self.catalog_url = catalog_url
        self.collection_name = collection_name
        self.assets = assets
        self.bbox = bbox
        self.time_interval = time_interval
        self.data_folder = data_folder
        self.limit = limit

    @property
    def url(self):
        return f"{self.catalog_url.rstrip('/')}/search"

    @property
    def payload(self):
        return dict(
            collections=[self.collection_name],
            bbox=[v for v in self.bbox],
            datetime="/".join(
                [f"{date.isoformat()}.000Z" for date in self.time_interval]
            ),
            limit=self.limit,
        )

    @property
    def headers(self):
        return {
            "Accept-Encoding": "gzip, deflate",
            "Accept": "*/*",
            "Connection": "keep-alive",
        }


class STACClient:
    def __init__(self, retry_count=3, backoff_factor=0.1):
        retry_strategy = Retry(
            total=retry_count,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.http = requests.Session()
        self.http.mount("https://", adapter)
        self.http.mount("http://", adapter)

    def download(
        self,
        reqs: List[STACSearchRequest],
        output_dir: str,
        max_threads: int = 5,
        timeout: int = 600,
    ) -> List[str]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            items_reqs = flatten(executor.map(self._get_items, reqs, timeout=timeout))
            assets = flatten(
                self._get_assets(item, filter_assets=req.assets)
                for item, req in items_reqs
            )

            download_asset_worker = partial(self._download_asset, output_dir=output_dir)
            files = executor.map(download_asset_worker, assets, timeout=timeout)

            items = [item for item, _ in items_reqs]
            return zip(list(items), list(files))

    def _get_items(self, request: STACSearchRequest) -> List[dict]:
        data = json.dumps(request.payload)
        LOGGER.debug(
            "POST %s with data=%s headers=%s", request.url, data, request.headers
        )
        res = self.http.post(
            request.url,
            data=data,
            headers=request.headers,
        )
        if res.status_code != 200:
            LOGGER.error("Response: %s", res.text)
        res.raise_for_status()
        json_body = res.json()
        return [(feat, request) for feat in json_body["features"]]

    def _get_assets(self, item: dict, filter_assets=None) -> List[dict]:
        return [
            v
            for k, v in item["assets"].items()
            if not filter_assets or k in filter_assets
        ]

    def _download_asset(self, asset: dict, *, output_dir: str) -> str:
        url = asset["href"]
        local_filename = url.split("/")[-1]
        local_path = os.path.join(output_dir, local_filename)
        if os.path.exists(local_path):
            LOGGER.info("File %s already exists. Skipping download.", local_path)
            return local_path
        LOGGER.info(f"Download {url} to {local_path} ({local_filename})")
        with self.http.get(url, stream=True) as r:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as f:
                shutil.copyfileobj(r.raw, f)
        return local_path
