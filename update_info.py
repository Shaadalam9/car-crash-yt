# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>

import csv
import os
from datetime import datetime, timedelta

import pandas as pd
from pytubefix import YouTube
from tqdm import tqdm

import common
from custom_logger import CustomLogger
from logmod import logs


# =====================================================
# Logging
# =====================================================

logs(
    show_level=common.get_configs("logger_level"),
    show_color=True,
)

logger = CustomLogger(__name__)


# =====================================================
# Configuration
# =====================================================

metadata_file = "mapping_metadata.csv"

csv_headers = [
    "id",
    "video",
    "title",
    "upload_date",
    "channel",
    "views",
    "description",
    "chapters",
    "segments",
    "date_updated",
]


# Try multiple YouTube clients.
#
# pytubefix currently defaults to ANDROID_VR.
# Some videos can incorrectly appear unavailable through
# one client while working through another.
YOUTUBE_CLIENTS = [
    "WEB",
    "MWEB",
    "TV",
    "IOS",
    "ANDROID_VR",
]


# =====================================================
# Helpers
# =====================================================

def safe_parse_video_list(value):
    """
    Parse the videos column without using ast or another parser.

    Expected examples:

        ['abc123', 'def456']

        ["abc123", "def456"]

        [abc123, def456]

    YouTube video IDs cannot contain commas, so splitting
    by comma is safe for this use case.
    """

    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    value = str(value).strip()

    if not value:
        return []

    if value.lower() in {
        "nan",
        "none",
        "<na>",
    }:
        return []

    # Remove surrounding brackets
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]

    if not value.strip():
        return []

    video_ids = []

    for item in value.split(","):

        item = (
            item
            .strip()
            .strip("'")
            .strip('"')
            .strip()
        )

        if not item:
            continue

        if item.lower() in {
            "nan",
            "none",
            "<na>",
        }:
            continue

        video_ids.append(item)

    return video_ids


def clean_text(value):
    """
    Convert metadata to safe text for CSV storage.
    """

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return (
        str(value)
        .replace("\n", " ")
        .replace("\r", " ")
        .strip()
    )


def is_missing(value):
    """
    Return True when a value should be regarded as missing.
    """

    if value is None:
        return True

    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass

    value = str(value).strip().lower()

    return value in {
        "",
        "nan",
        "none",
        "<na>",
    }


def extract_chapters(yt, video_id):
    """
    Extract chapters without failing the entire video
    if chapter extraction itself has a problem.
    """

    chapters = []

    try:

        youtube_chapters = yt.chapters

        if not youtube_chapters:
            return chapters

        for chapter in youtube_chapters:

            try:

                chapters.append(
                    {
                        "title": clean_text(
                            chapter.title
                        ),
                        "timestamp": str(
                            timedelta(
                                seconds=chapter.start_seconds
                            )
                        ),
                    }
                )

            except Exception as error:

                logger.info(
                    "⚠️ failed to extract one chapter "
                    "for {}: {}",
                    video_id,
                    error,
                )

    except Exception as error:

        logger.info(
            "⚠️ failed to extract chapters "
            "for {}: {}",
            video_id,
            error,
        )

    return chapters


# =====================================================
# YouTube metadata fetching
# =====================================================

def fetch_with_client(
    video_id,
    client,
    use_oauth=False,
):
    """
    Try fetching one video using one pytubefix client.

    Returns metadata dict on success.
    Raises an exception on failure.
    """

    url = (
        "https://www.youtube.com/watch?v="
        f"{video_id}"
    )

    yt = YouTube(
        url,
        client=client,
        use_oauth=use_oauth,
        allow_oauth_cache=True,
    )

    # Accessing these properties forces pytubefix
    # to fetch and validate video metadata.
    title = yt.title
    description = yt.description
    channel = yt.channel_id
    views = yt.views
    publish_date = yt.publish_date

    if publish_date:
        upload_date = publish_date.strftime(
            "%d%m%Y"
        )
    else:
        upload_date = ""

    chapters = extract_chapters(
        yt,
        video_id,
    )

    return {
        "video": str(video_id),

        "title": clean_text(
            title
        ),

        "upload_date": clean_text(
            upload_date
        ),

        "channel": clean_text(
            channel
        ),

        # Intentionally stored as text.
        #
        # Do not force this column to int64 because
        # missing metadata must remain possible.
        "views": clean_text(
            views
        ),

        "description": clean_text(
            description
        ),

        "chapters": clean_text(
            chapters
        ),
    }


def get_video_info(video_id):
    """
    Fetch metadata for one video using several pytubefix
    clients.

    Strategy:

        WEB
         ↓
        MWEB
         ↓
        TV
         ↓
        IOS
         ↓
        ANDROID_VR
         ↓
        OAuth fallback

    A failure from one client does not mean the video
    is genuinely unavailable.
    """

    video_id = str(video_id).strip()

    # -------------------------------------------------
    # First try normal public access
    # -------------------------------------------------

    for client in YOUTUBE_CLIENTS:

        try:

            info = fetch_with_client(
                video_id=video_id,
                client=client,
                use_oauth=False,
            )

            safe_title = (
                info["title"]
                .replace("{", "{{")
                .replace("}", "}}")
            )

            safe_channel = (
                info["channel"]
                .replace("{", "{{")
                .replace("}", "}}")
            )

            logger.info(
                "✅ fetched: {} | "
                "client: {} | "
                "title: {} | "
                "upload: {} | "
                "channel: {} | "
                "views: {}",
                video_id,
                client,
                safe_title,
                info["upload_date"],
                safe_channel,
                info["views"],
            )

            return info

        except Exception as error:

            logger.info(
                "⚠️ client {} failed for {}: {}",
                client,
                video_id,
                error,
            )

    # -------------------------------------------------
    # OAuth fallback
    #
    # Useful for content where an authenticated YouTube
    # session is required.
    # -------------------------------------------------

    logger.info(
        "🔐 trying OAuth fallback for {}",
        video_id,
    )

    oauth_clients = [
        "WEB",
        "TV",
        "ANDROID_VR",
    ]

    for client in oauth_clients:

        try:

            info = fetch_with_client(
                video_id=video_id,
                client=client,
                use_oauth=True,
            )

            safe_title = (
                info["title"]
                .replace("{", "{{")
                .replace("}", "}}")
            )

            safe_channel = (
                info["channel"]
                .replace("{", "{{")
                .replace("}", "}}")
            )

            logger.info(
                "✅ fetched with OAuth: {} | "
                "client: {} | "
                "title: {} | "
                "upload: {} | "
                "channel: {} | "
                "views: {}",
                video_id,
                client,
                safe_title,
                info["upload_date"],
                safe_channel,
                info["views"],
            )

            return info

        except Exception as error:

            logger.info(
                "⚠️ OAuth client {} failed "
                "for {}: {}",
                client,
                video_id,
                error,
            )

    # -------------------------------------------------
    # All clients failed
    # -------------------------------------------------

    logger.info(
        "❌ all clients failed for {}",
        video_id,
    )

    return None


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    # =================================================
    # Load mapping CSV
    # =================================================

    mapping_csv = common.get_configs(
        "MAPPING_CSV"
    )

    df_map = pd.read_csv(
        mapping_csv,
        dtype=str,
        keep_default_na=False,
    )

    if "videos" not in df_map.columns:
        raise ValueError(
            "The mapping CSV does not contain "
            "a 'videos' column."
        )

    # =================================================
    # Collect video IDs
    # =================================================

    all_video_ids = []

    for row in df_map["videos"]:

        videos = safe_parse_video_list(
            row
        )

        for video_id in videos:

            video_id = (
                str(video_id)
                .strip()
            )

            if not video_id:
                continue

            all_video_ids.append(
                video_id
            )

    if all_video_ids:

        video_count = (
            pd.Series(
                all_video_ids,
                dtype=str,
            )
            .value_counts()
            .to_dict()
        )

    else:

        video_count = {}

    unique_video_ids = list(
        video_count.keys()
    )

    logger.info(
        "🎬 found {} unique videos "
        "in mapping CSV",
        len(unique_video_ids),
    )

    # =================================================
    # Load metadata file
    # =================================================

    if os.path.exists(metadata_file):

        # Important:
        #
        # Load all metadata as strings.
        #
        # This avoids pandas deciding that "views" is
        # int64 and later refusing an empty value.
        existing_df = pd.read_csv(
            metadata_file,
            dtype=str,
            keep_default_na=False,
        )

        # ---------------------------------------------
        # Add missing columns if metadata CSV is old
        # ---------------------------------------------

        for col in csv_headers:

            if col not in existing_df.columns:
                existing_df[col] = ""

        # Keep canonical ordering
        existing_df = existing_df[
            csv_headers
        ].copy()

        # ---------------------------------------------
        # ID is the only numeric column we require
        # ---------------------------------------------

        existing_df["id"] = (
            pd.to_numeric(
                existing_df["id"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )

        # Everything else remains object/string compatible
        for col in csv_headers:

            if col == "id":
                continue

            existing_df[col] = (
                existing_df[col]
                .astype("object")
            )

        # ---------------------------------------------
        # Normalise video IDs
        # ---------------------------------------------

        existing_df["video"] = (
            existing_df["video"]
            .astype(str)
            .str.strip()
        )

        existing_ids = {
            video_id
            for video_id
            in existing_df["video"].tolist()
            if not is_missing(video_id)
        }

        if existing_df.empty:
            last_id = 0
        else:
            last_id = int(
                existing_df["id"].max()
            )

        logger.info(
            "🗃️ found {} videos "
            "in existing metadata",
            len(existing_ids),
        )

        # =================================================
        # Detect incomplete existing metadata
        # =================================================

        critical_cols = [
            "title",
            "upload_date",
            "channel",
            "views",
        ]

        failed_mask = (
            existing_df[
                critical_cols
            ]
            .apply(
                lambda column:
                column.map(is_missing)
            )
            .any(axis=1)
        )

        failed_ids = {
            str(video_id).strip()
            for video_id
            in existing_df.loc[
                failed_mask,
                "video",
            ].tolist()
            if not is_missing(video_id)
        }

        logger.info(
            "🔁 found {} videos "
            "with missing metadata",
            len(failed_ids),
        )

    else:

        # =================================================
        # Create a new metadata CSV
        # =================================================

        with open(
            metadata_file,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(
                file,
                quoting=csv.QUOTE_MINIMAL,
                escapechar="\\",
            )

            writer.writerow(
                csv_headers
            )

        existing_df = pd.DataFrame(
            columns=csv_headers
        )

        existing_df["id"] = pd.Series(
            dtype="int64"
        )

        existing_ids = set()
        failed_ids = set()
        last_id = 0

        logger.info(
            "📁 created new metadata file"
        )

    # =================================================
    # Determine what needs to be fetched
    # =================================================

    missing_in_metadata = [
        video_id
        for video_id
        in unique_video_ids
        if video_id not in existing_ids
    ]

    retry_videos = [
        video_id
        for video_id
        in unique_video_ids
        if video_id in failed_ids
    ]

    # Remove accidental duplicates while preserving order
    videos_to_fetch = list(
        dict.fromkeys(
            missing_in_metadata
            + retry_videos
        )
    )

    logger.info(
        "🔍 total videos to fetch: {} "
        "({} missing, {} need update)",
        len(videos_to_fetch),
        len(missing_in_metadata),
        len(retry_videos),
    )

    # =================================================
    # Current date
    # =================================================

    now = datetime.now().strftime(
        "%d%m%Y"
    )

    current_id = last_id

    # =================================================
    # Main fetch loop
    # =================================================

    for video_id in tqdm(
        videos_to_fetch
    ):

        video_id = (
            str(video_id)
            .strip()
        )

        # ---------------------------------------------
        # Fetch YouTube metadata
        # ---------------------------------------------

        info = get_video_info(
            video_id
        )

        # ---------------------------------------------
        # Fetch completely failed
        # ---------------------------------------------

        if info is None:

            logger.info(
                "⏭️ skipping update for {} "
                "because metadata could not "
                "be fetched using any client",
                video_id,
            )

            # Do not write empty metadata.
            #
            # If this was already in metadata.csv,
            # its existing values remain untouched.
            #
            # If it was a new video, it remains absent
            # and will therefore be retried next time.
            continue

        # =================================================
        # Dataset specific fields
        # =================================================

        info["segments"] = clean_text(
            video_count.get(
                video_id,
                1,
            )
        )

        info["date_updated"] = now

        # =================================================
        # Existing or new row?
        # =================================================

        mask = (
            existing_df["video"]
            == video_id
        )

        if mask.any():

            # -----------------------------------------
            # Existing video
            # -----------------------------------------

            existing_id = int(
                existing_df.loc[
                    mask,
                    "id",
                ].iloc[0]
            )

            info["id"] = existing_id

            # -----------------------------------------
            # Update metadata
            # -----------------------------------------

            for col in csv_headers:

                if col not in info:
                    continue

                value = info[col]

                # ID is numeric
                if col == "id":

                    existing_df.loc[
                        mask,
                        col,
                    ] = int(value)

                else:

                    existing_df.loc[
                        mask,
                        col,
                    ] = clean_text(
                        value
                    )

            logger.info(
                "🔄 refreshed existing metadata "
                "for {}",
                video_id,
            )

        else:

            # -----------------------------------------
            # New video
            # -----------------------------------------

            current_id += 1

            info["id"] = current_id

            # Make sure every expected column exists
            row_data = {}

            for col in csv_headers:

                if col == "id":

                    row_data[col] = int(
                        info["id"]
                    )

                else:

                    row_data[col] = clean_text(
                        info.get(
                            col,
                            "",
                        )
                    )

            new_row = pd.DataFrame(
                [row_data],
                columns=csv_headers,
            )

            existing_df = pd.concat(
                [
                    existing_df,
                    new_row,
                ],
                ignore_index=True,
            )

            logger.info(
                "➕ added new metadata "
                "for {}",
                video_id,
            )

        # =================================================
        # Keep ID numeric only
        # =================================================

        existing_df["id"] = (
            pd.to_numeric(
                existing_df["id"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )

        # =================================================
        # Keep video ID normalised
        # =================================================

        existing_df["video"] = (
            existing_df["video"]
            .astype(str)
            .str.strip()
        )

        # =================================================
        # Save after every successful video
        #
        # This means progress is preserved even if the
        # program is interrupted later.
        # =================================================

        existing_df.to_csv(
            metadata_file,
            index=False,
        )

        logger.info(
            "💾 updated metadata for {}",
            video_id,
        )

    # =================================================
    # Final save
    # =================================================

    existing_df.to_csv(
        metadata_file,
        index=False,
    )

    logger.info(
        "✅ all videos processed "
        "and metadata file updated."
    )
