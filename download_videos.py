# by Shadab Alam <md_shadab_alam@outlook.com> and Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
# -----------------------------------------------------------------------------
# Pipeline overview:
# - Reads a mapping CSV describing video IDs.
# - Goes through all videos listed in the mapping file.
# - Skips videos that already exist in any configured video directory.
# - Downloads missing videos from YouTube.
# -----------------------------------------------------------------------------

import os
from helper_script import Youtube_Helper
import pandas as pd
from custom_logger import CustomLogger
from logmod import logs
import common

# Configure logging based on config file (verbosity & ANSI colors)
logs(show_level=common.get_configs("logger_level"), show_color=True)
logger = CustomLogger(__name__)
helper = Youtube_Helper()


def video_already_downloaded(vid: str, video_paths) -> bool:
    """
    Check whether a video already exists in any configured video directory.

    Parameters:
        vid (str): YouTube video ID.
        video_paths (list): Configured video directories.

    Returns:
        bool: True if the video already exists, otherwise False.
    """
    for video_path in video_paths:
        video_file_path = os.path.join(video_path, f"{vid}.mp4")

        if os.path.isfile(video_file_path):
            logger.info(f"{vid}: video already downloaded at {video_file_path}. Skipping.")
            return True

    return False


def download_videos_from_mapping(mapping, video_paths) -> int:
    """
    Go through the mapping file and download videos that are not already present.

    Parameters:
        mapping (pd.DataFrame): Mapping dataframe containing the videos column.
        video_paths (list): Configured video directories.

    Returns:
        int: Number of videos successfully downloaded.
    """
    if not video_paths:
        logger.error("No video directories configured.")
        return 0

    output_path = video_paths[-1]
    os.makedirs(output_path, exist_ok=True)

    videos_seen = set()
    counter_downloaded = 0

    # ------------------------------------------------------------------
    # Go through mapping rows and collect unique video IDs.
    # ------------------------------------------------------------------
    for _, row in mapping.iterrows():
        try:
            video_ids = [
                v.strip()
                for v in str(row["videos"]).strip("[]").split(",")
                if v.strip()
            ]

            for vid in video_ids:

                # Avoid checking/downloading the same video multiple times
                # when it appears more than once in mapping.csv.
                if vid in videos_seen:
                    continue

                videos_seen.add(vid)

                # ------------------------------------------------------
                # Skip videos that already exist.
                # ------------------------------------------------------
                if video_already_downloaded(vid, video_paths):
                    continue

                # ------------------------------------------------------
                # Download missing video.
                # ------------------------------------------------------
                logger.info(f"{vid}: video not found locally. Starting download.")

                result = helper.download_video_with_resolution(
                    vid,
                    output_path=output_path,
                )

                if result is None:
                    logger.error(f"{vid}: video download failed.")
                    continue

                video_file_path, video_title, resolution, fps = result

                logger.info(
                    f"{vid}: download completed. "
                    f"Resolution={resolution}, FPS={fps}, "
                    f"path={video_file_path}."
                )

                counter_downloaded += 1

        except Exception as e:
            logger.warning(f"Skipping malformed mapping row due to error: {e!r}")
            continue

    return counter_downloaded


# =============================================================================
# Main entry point
# =============================================================================
if __name__ == "__main__":
    try:
        # ---------------------------------------------------------------------
        # Load configuration
        # ---------------------------------------------------------------------
        mapping_path = common.get_configs("MAPPING_CSV")
        video_paths = common.get_configs("videos")

        # ---------------------------------------------------------------------
        # Read mapping
        # ---------------------------------------------------------------------
        mapping = pd.read_csv(mapping_path)

        logger.info(f"Loaded mapping file: {mapping_path}")
        logger.info(f"Mapping contains {len(mapping)} rows.")

        # ---------------------------------------------------------------------
        # Download missing videos
        # ---------------------------------------------------------------------
        counter_downloaded = download_videos_from_mapping(
            mapping,
            video_paths,
        )

        logger.info(
            f"Download pass completed. "
            f"{counter_downloaded} new video(s) downloaded."
        )

    except Exception as e:
        logger.error(f"Video download pipeline failed: {e!r}")
        raise