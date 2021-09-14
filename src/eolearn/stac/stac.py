"""This module contains STAC related classes and functions."""

__author__ = "Damián Silvani"
__copyright__ = "Dymaxion Labs"
__license__ = "MIT"

import concurrent.futures
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


class STACItemsRequest:
    def __init__(
        self,
        bbox=None,
        time_interval=None,
        data_folder=None,
        assets=None,
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

    @property
    def url(self):
        return f"{self.catalog_url}/collections/{self.collection_name}/items"

    @property
    def parameters(self):
        return dict(
            bbox=",".join(str(v) for v in self.bbox),
            datetime=",".join(
                [f"{date.isoformat()}.000Z" for date in self.time_interval]
            ),
        )


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
        reqs: List[STACItemsRequest],
        output_dir: str,
        max_threads: int = 5,
        timeout: int = 360,
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

    def _get_items(self, request: STACItemsRequest) -> List[dict]:
        res = self.http.get(request.url, params=request.parameters).json()
        return [(feat, request) for feat in res["features"]]

    def _get_assets(self, item: dict, filter_assets=None) -> List[dict]:
        assets = item["assets"]
        if filter_assets:
            assets = [v for k, v in assets.items() if k in filter_assets]
        return assets

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
