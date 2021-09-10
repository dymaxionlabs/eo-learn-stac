"""This module contains EOTask subclasses, like STACInputTask."""

__author__ = "Damián Silvani"
__copyright__ = "Dymaxion Labs"
__license__ = "MIT"


import logging
from itertools import groupby

import rasterio
from eolearn.core import EOPatch, EOTask
from sentinelhub import parse_time_interval

from .stac import STACClient, STACItemsRequest

LOGGER = logging.getLogger(__name__)


class STACInputTask(EOTask):
    def __init__(
        self,
        catalog_url,
        assets=None,
        subdataset=None,
        bands=None,
        cache_folder=None,
        max_threads=None,
        single_scene=True,
        mosaicking_order="mostRecent",
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
        :param cache_folder: Path to cache_folder. If set to None (default) requests will not be cached.
        :type cache_folder: str
        :param max_threads: Maximum threads to be used when downloading data.
        :type max_threads: int
        :param single_scene: If true, the service will compute a single image for the given time interval using mosaicking.
        :type single_scene: bool
        :param mosaicking_order: Mosaicking order, which has to be either "mostRecent" or "leastRecent"
        :type mosaicking_order: str
        """
        self.catalog_url = catalog_url
        self.collection_name = collection_name
        self.assets = assets
        self.subdataset = subdataset
        self.bands = bands
        self.cache_folder = cache_folder
        self.max_threads = max_threads
        self.single_scene = single_scene
        self.mosaicking_order = mosaicking_order

    def execute(self, eopatch=None, bbox=None, time_interval=None):
        """Main execute method for downloading and clipping image"""

        eopatch = eopatch or EOPatch()

        self._check_and_set_eopatch_bbox(bbox, eopatch)

        reqs = self._build_requests(eopatch.bbox, eopatch.timestamp, time_interval)

        LOGGER.debug(
            "Downloading %d requests of catalog %s",
            len(reqs),
            str(self.catalog_url),
        )
        client = STACClient()
        assets_and_files = list(
            client.download(
                reqs, output_dir=self.cache_folder, max_threads=self.max_threads
            )
        )
        LOGGER.debug("Downloads complete")

        # Group assets by date
        assets_by_date = {
            k: list(v)
            for k, v in groupby(
                assets_and_files, lambda x: x[0]["properties"]["datetime"]
            )
        }

        if not assets_by_date:
            return

        # Open an image to get pixel size
        first_date = list(assets_by_date.values())[0]
        _, first_image_path = first_date[0]
        LOGGER.info(f"Get resolution from first image: {first_image_path}")
        size_x, size_y = self._get_pixel_size(first_image_path)

        # # Calculate eopatch shape from number of dates and image size in pixels
        temporal_dim = len(assets_by_date)
        shape = temporal_dim, size_y, size_x
        LOGGER.info(f"Shape: {shape}")

        # TODO: Extract specified subdatasets and bands
        # if self.single_scene:
        #     # TODO: Mosaic images
        #     pass

        # Clip mosaics to eopatch bbox
        # self._extract_data(eopatch, assets_by_date, shape)

        eopatch.meta_info["size_x"] = size_x
        eopatch.meta_info["size_y"] = size_y
        # if timestamp:  # do not overwrite time interval in case of timeless features
        #     eopatch.meta_info["time_interval"] = time_interval

        # self._add_meta_info(eopatch)

        return eopatch

    def _get_pixel_size(self, path):
        if self.subdataset:
            path = self._add_subdataset_to_path(path)
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

    def _extract_data(self, eopatch, files, shape):
        """Extract data from the received images and assign them to eopatch features"""
        pass

    def _build_requests(self, bbox, timestamp, time_interval):
        """Build requests"""
        if timestamp:
            dates = [
                (date - self.time_difference, date + self.time_difference)
                for date in timestamp
            ]
        else:
            dates = (
                [parse_time_interval(time_interval, allow_undefined=True)]
                if time_interval
                else [None]
            )

        return [self._create_request(date, bbox) for date in dates]

    def _create_request(self, time_interval, bbox):
        """Create an instance of Request"""
        return STACItemsRequest(
            catalog_url=self.catalog_url,
            collection_name=self.collection_name,
            assets=self.assets,
            bbox=bbox,
            time_interval=time_interval,
        )
