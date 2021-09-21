"""This module contains EOTask subclasses, like STACInputTask."""

__author__ = "Damián Silvani"
__copyright__ = "Dymaxion Labs"
__license__ = "MIT"


import datetime
import json
import logging
import os
import tempfile
from itertools import groupby

import numpy as np
import rasterio
import rasterio.errors
import rasterio.mask
import rasterio.merge
import rasterio.warp
from sentinelhub import parse_time_interval

from eolearn.core import EOPatch, EOTask, FeatureType

from .stac import STACClient, STACSearchRequest

LOGGER = logging.getLogger(__name__)


class STACInputTask(EOTask):
    def __init__(
        self,
        catalog_url,
        assets=None,
        subdataset=None,
        bands=None,
        all_touched=True,
        max_threads=None,
        retry_count=3,
        backoff_factor=0.1,
        limit=10,
        *,
        collection_name,
        # size,
    ):
        """
        :param catalog_url: A STAC catalog URL
        :type catalog_url: str
        :param collection_name: The name of the collection to be downloaded
        :type collection_name: str
        :param assets: A list of asset names to download. If empty or None, all assets are downloaded.
        :type assets: iterable of str
        :param subdataset: A subdataset to extract from, if any.
        :type subdataset: str
        :param bands: A list of bands to extract from. If empty or None, all bands are extracted.
        :type bands: iterable of int
        :param size: Number of pixels in x and y dimension.
        :type size: tuple(int, int)
        :param all_touched: If True, all pixels touched by the geometry are extracted.
        :type all_touched: bool
        :param max_threads: Maximum threads to be used when downloading data.
        :type max_threads: int
        :param retry_count: Number of retries when downloading data.
        :type retry_count: int
        :param backoff_factor: Backoff factor for retries.
        :type backoff_factor: float
        :param limit: Maximum number of assets to be downloaded.
        :type limit: int
        """
        self.catalog_url = catalog_url
        self.collection_name = collection_name
        self.assets = assets
        self.subdataset = subdataset
        self.bands = bands
        self.all_touched = all_touched
        self.max_threads = max_threads
        self.retry_count = retry_count
        self.backoff_factor = backoff_factor
        self.limit = limit

    def execute(self, eopatch=None, bbox=None, time_interval=None, cache_folder=None):
        """Main execute method for downloading and clipping image"""

        eopatch = eopatch or EOPatch()
        self._check_and_set_eopatch_bbox(bbox, eopatch)

        with tempfile.TemporaryDirectory(prefix="eo-learn-stac_") as tmpdir:
            cache_folder = tmpdir if cache_folder is None else cache_folder

            reqs = self._build_requests(eopatch.bbox, eopatch.timestamp, time_interval)

            LOGGER.info(
                "Downloading %d requests of catalog %s",
                len(reqs),
                str(self.catalog_url),
            )
            client = STACClient(
                retry_count=self.retry_count, backoff_factor=self.backoff_factor
            )
            raw_dir = os.path.join(cache_folder, "raw")
            assets_and_files = list(
                client.download(reqs, output_dir=raw_dir, max_threads=self.max_threads)
            )
            LOGGER.info("Downloads complete")

            # Extract subdataset (if needed)
            extracted_dir = os.path.join(cache_folder, "extracted")
            assets_and_files = self._extract_subdataset(
                assets_and_files, output_dir=extracted_dir
            )

            # Group assets by date
            assets_by_date = {
                datetime.datetime.strptime(k, "%Y-%m-%dT%H:%M:%S.%fZ"): list(v)
                for k, v in groupby(
                    assets_and_files,
                    lambda x: x[0]["properties"]["datetime"],
                )
            }

            # Filter assets by time interval (just in case)
            if time_interval:
                min_dt, max_dt = parse_time_interval(time_interval)
                assets_by_date = {
                    k: v for k, v in assets_by_date.items() if min_dt <= k <= max_dt
                }

            if not assets_by_date:
                LOGGER.warn("No assets found")
                return eopatch

            # Merge assets by date into a single image
            merged_dir = os.path.join(cache_folder, "merged")
            image_by_date = self._merge_images(assets_by_date, output_dir=merged_dir)

            # Clip mosaics to eopatch bbox, set timestamp and shape on meta_info
            self._extract_data(eopatch, image_by_date)

            self._add_meta_info(eopatch, list(image_by_date.values()))

        return eopatch

    def _get_pixel_size(self, path):
        with rasterio.open(path) as src:
            return src.res

    def _add_subdataset_to_path(self, path):
        return self.subdataset.replace("%s", '"%s"') % path.replace('"', '\\"')

    @staticmethod
    def _check_and_set_eopatch_bbox(bbox, eopatch):
        if eopatch.bbox is None:
            if bbox is None:
                raise ValueError(
                    "Either the eopatch or the task must provide valid bbox."
                )
            eopatch.bbox = bbox
            return

        if bbox is None or eopatch.bbox == bbox:
            return
        raise ValueError(
            "Either the eopatch or the task must provide bbox, or they must be the same."
        )

    def _extract_subdataset(self, assets_and_files, *, output_dir):
        """Extract subdataset and bands from assets"""
        if not self.subdataset:
            return assets_and_files

        os.makedirs(output_dir, exist_ok=True)
        res = []
        for asset, src_path in assets_and_files:
            name, _ = os.path.splitext(os.path.basename(src_path))
            dst_path = os.path.join(output_dir, f"{name}.tif")
            res.append((asset, dst_path))

            if os.path.exists(dst_path):
                continue

            subdataset_path = self._add_subdataset_to_path(src_path)
            with rasterio.open(subdataset_path) as src:
                profile = src.profile.copy()
                profile.update(driver="GeoTIFF")
                with rasterio.open(dst_path, "w", **src.profile) as dst:
                    bands = (
                        range(1, src.count + 1) if self.bands is None else self.bands
                    )
                    for b in bands:
                        for _, window in src.block_windows(b):
                            data = src.read(b, window=window)
                            dst.write(data, b, window=window)

        return res

    def _merge_images(self, assets_by_date, method="first", *, output_dir):
        """Merge images from assets_by_date into a single image"""
        image_by_date = {}
        for i, (date, assets) in enumerate(assets_by_date.items()):
            dst_path = os.path.join(output_dir, f'{i}_{date.strftime("%Y%m%d")}.tif')
            datasets = [rasterio.open(path) for _, path in assets]
            os.makedirs(output_dir, exist_ok=True)
            rasterio.merge.merge(datasets, dst_path=dst_path, method=method)
            LOGGER.info("%s written", dst_path)
            for ds in datasets:
                ds.close()
            image_by_date[date] = (assets, dst_path)
        return image_by_date

    def _extract_data(self, eopatch, image_by_date):
        """
        Extract data from the received images and assign them as BANDS data
        feature on EOPatch.

        """
        data = []
        timestamp = []
        for datetime, (_, file) in image_by_date.items():
            with rasterio.open(file) as src:
                geom = eopatch.bbox.get_geojson()
                dst_crs = eopatch.bbox.crs.ogc_string()

                geom_src = rasterio.warp.transform_geom(dst_crs, src.crs, geom)

                out_data, _ = rasterio.mask.mask(
                    src, [geom_src], crop=True, all_touched=self.all_touched
                )
                out_data = np.dstack(out_data)
                data.append(out_data)
                timestamp.append(datetime)

        bands = np.array(data)

        eopatch.timestamp = timestamp
        eopatch[(FeatureType.DATA, "BANDS")] = bands

        size_x, size_y = bands.shape[2], bands.shape[1]
        eopatch.meta_info["size_x"] = size_x
        eopatch.meta_info["size_y"] = size_y

    def _build_requests(self, bbox, timestamp, time_interval):
        """Build requests"""
        if timestamp:
            dates = timestamp
        else:
            dates = (
                [parse_time_interval(time_interval, allow_undefined=True)]
                if time_interval
                else [None]
            )
        return [self._create_request(date, bbox) for date in dates]

    def _create_request(self, time_interval, bbox):
        """Create an instance of Request"""
        return STACSearchRequest(
            catalog_url=self.catalog_url,
            collection_name=self.collection_name,
            assets=self.assets,
            bbox=bbox,
            time_interval=time_interval,
            limit=self.limit,
        )

    def _add_meta_info(self, eopatch, assets):
        """Add meta info to eopatch"""
        eopatch.meta_info["stac_items"] = json.dumps(assets)
