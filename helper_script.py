# by Shadab Alam <md_shadab_alam@outlook.com> and Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>

import os
import re
import datetime
import shutil
import subprocess
import sys
import logging

from pytubefix import YouTube
from pytubefix.cli import on_progress
import cv2
import yt_dlp

from custom_logger import CustomLogger
import common

logger = CustomLogger(__name__)


# logging of attempts to upgrade packages
UPGRADE_LOG_FILE = "upgrade_log.json"


class Youtube_Helper:
    """
    A helper class for managing YouTube video downloads.

    Features:
        - Downloads videos via pytubefix.
        - Falls back to yt-dlp if pytubefix fails.
        - Selects a preferred resolution up to 720p.
        - Retrieves video FPS.
    """

    def __init__(self, video_title=None):
        """
        Initialises a new instance of the class.

        Parameters:
            video_title (str, optional): The title of the video.
        """
        self.video_title = video_title
        self.update_package = common.get_configs("update_package")
        self.need_authentication = common.get_configs("need_authentication")
        self.client = common.get_configs("client")

    def set_video_title(self, title):
        """
        Sets the video title for the instance.

        Parameters:
            title (str): The new title for the video.
        """
        self.video_title = title

    def load_upgrade_log(self):
        """
        Load package upgrade attempt log from file.

        Returns:
            dict: Dictionary with package names and last upgrade date.
        """
        if not os.path.exists(UPGRADE_LOG_FILE):
            return {}

        try:
            import json

            with open(UPGRADE_LOG_FILE, "r") as file:
                return json.load(file)
        except Exception:
            return {}

    def save_upgrade_log(self, log_data):
        """
        Save package upgrade attempt log.

        Parameters:
            log_data (dict): Dictionary containing package upgrade dates.
        """
        import json

        with open(UPGRADE_LOG_FILE, "w") as file:
            json.dump(log_data, file)

    def was_upgraded_today(self, package_name):
        """
        Check whether the given package was already upgraded today.

        Parameters:
            package_name (str): Name of the package.

        Returns:
            bool: True if upgraded today, False otherwise.
        """
        log_data = self.load_upgrade_log()
        today = datetime.date.today().isoformat()

        return log_data.get(package_name) == today

    def mark_as_upgraded(self, package_name):
        """
        Mark a package as upgraded by saving today's date.

        Parameters:
            package_name (str): Name of the package.
        """
        log_data = self.load_upgrade_log()
        log_data[package_name] = datetime.date.today().isoformat()
        self.save_upgrade_log(log_data)

    def upgrade_package_if_needed(self, package_name: str) -> None:
        """
        Upgrade a Python package using uv once per day.

        Parameters:
            package_name (str): Name of the package.
        """
        if self.was_upgraded_today(package_name):
            logging.debug(
                "%s upgrade already attempted today. Skipping.",
                package_name,
            )
            return

        uv_exe = Youtube_Helper._resolve_uv_executable()

        if not uv_exe:
            logging.error(
                "Cannot upgrade %s because `uv` was not found.",
                package_name,
            )
            self.mark_as_upgraded(package_name)
            return

        cmd = [
            uv_exe,
            "pip",
            "install",
            "--upgrade",
            package_name,
            "--python",
            sys.executable,
        ]

        try:
            logging.info(
                "Upgrading %s with uv (targeting %s)...",
                package_name,
                sys.executable,
            )

            subprocess.check_call(cmd)

            logging.info("%s upgraded successfully.", package_name)
            self.mark_as_upgraded(package_name)

        except subprocess.CalledProcessError as e:
            logging.error(
                "Failed to upgrade %s with uv: %s",
                package_name,
                e,
            )

            self.mark_as_upgraded(package_name)

    @staticmethod
    def _resolve_uv_executable() -> str | None:
        """
        Resolve the uv executable in a cross-platform manner.

        Returns:
            str | None: Absolute path to uv if found.
        """
        uv_on_path = shutil.which("uv")

        if uv_on_path:
            return uv_on_path

        candidates: list[str] = []

        home = os.path.expanduser("~")

        candidates.append(
            os.path.join(home, ".local", "bin", "uv")
        )

        candidates.append(
            os.path.join(home, ".local", "bin", "uv.exe")
        )

        appdata = os.environ.get("APPDATA")

        if appdata:
            candidates.append(
                os.path.join(appdata, "uv", "uv.exe")
            )

        py_dir = os.path.dirname(sys.executable)

        candidates.append(
            os.path.join(py_dir, "uv")
        )

        candidates.append(
            os.path.join(py_dir, "uv.exe")
        )

        for path in candidates:
            if (
                path
                and os.path.isfile(path)
                and os.access(path, os.X_OK)
            ):
                return path

        return None

    def download_video_with_resolution(
        self,
        vid,
        resolutions=["720p", "480p", "360p", "144p"],
        output_path=".",
    ):
        """
        Downloads a YouTube video in one of the specified resolutions and
        returns video details.

        Uses pytubefix first, then yt-dlp as fallback.

        Downloads into output_path/.ftp_tmp first, then moves the completed
        file to output_path.
        """
        selected_resolution = None

        temp_dir = os.path.join(output_path, ".ftp_tmp")
        final_video_file_path = os.path.join(
            output_path,
            f"{vid}.mp4",
        )
        temp_video_file_path = os.path.join(
            temp_dir,
            f"{vid}.mp4",
        )

        os.makedirs(output_path, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)

        # ------------------------------------------------------------------
        # Do not download an existing video again.
        # ------------------------------------------------------------------
        if os.path.isfile(final_video_file_path):
            logger.info(
                f"{vid}: video already exists at "
                f"{final_video_file_path}. Skipping."
            )
            return None

        try:
            if (
                self.update_package
                and datetime.datetime.today().weekday() == 0
            ):
                self.upgrade_package_if_needed("pytubefix")

            youtube_url = (
                f"https://www.youtube.com/watch?v={vid}"
            )

            if self.need_authentication:
                youtube_object = YouTube(
                    youtube_url,
                    self.client,
                    use_oauth=True,
                    allow_oauth_cache=True,
                    on_progress_callback=on_progress,
                )
            else:
                youtube_object = YouTube(
                    youtube_url,
                    self.client,
                    on_progress_callback=on_progress,
                )

            selected_stream = None
            selected_resolution = None

            # --------------------------------------------------------------
            # 1) Preferred resolutions
            # --------------------------------------------------------------
            for resolution in resolutions:
                video_streams = youtube_object.streams.filter(
                    res=resolution
                )

                if video_streams:
                    selected_resolution = resolution

                    logger.debug(
                        f"Found video {vid} in {resolution}."
                    )

                    selected_stream = (
                        video_streams.first()
                        if hasattr(video_streams, "first")
                        else video_streams[0]
                    )

                    break

            # --------------------------------------------------------------
            # 2) Fallback logic if no preferred match
            # --------------------------------------------------------------
            if not selected_stream:

                def _height(s) -> int:
                    h = getattr(s, "height", None)

                    if isinstance(h, int) and h > 0:
                        return h

                    res_attr = (
                        getattr(s, "resolution", None)
                        or getattr(s, "res", None)
                        or ""
                    )

                    match = re.search(
                        r"(\d{3,4})p",
                        str(res_attr),
                    )

                    return (
                        int(match.group(1))
                        if match
                        else -1
                    )

                def _is_mp4(s) -> bool:
                    mime = (
                        getattr(s, "mime_type", "")
                        or ""
                    )

                    return "mp4" in mime.lower()

                def _pick_prefer_progressive(
                    streams_at_height,
                ):
                    progressive = [
                        s
                        for s in streams_at_height
                        if getattr(
                            s,
                            "is_progressive",
                            False,
                        )
                    ]

                    return (
                        progressive[0]
                        if progressive
                        else streams_at_height[0]
                    )

                streams = list(youtube_object.streams)

                # ----------------------------------------------------------
                # Highest available <= 720p
                # ----------------------------------------------------------
                le_720 = [
                    s
                    for s in streams
                    if _is_mp4(s)
                    and 0 < _height(s) <= 720
                ]

                if le_720:
                    max_h = max(
                        _height(s)
                        for s in le_720
                    )

                    at_max_h = [
                        s
                        for s in le_720
                        if _height(s) == max_h
                    ]

                    chosen = _pick_prefer_progressive(
                        at_max_h
                    )

                    selected_stream = chosen

                    selected_resolution = (
                        getattr(
                            chosen,
                            "resolution",
                            None,
                        )
                        or getattr(
                            chosen,
                            "res",
                            None,
                        )
                        or f"{max_h}p"
                    )

                    logger.debug(
                        f"{vid}: no preferred match; "
                        f"picked highest available "
                        f"{selected_resolution} (≤720p)."
                    )

                else:
                    # ------------------------------------------------------
                    # Lowest available > 720p
                    # ------------------------------------------------------
                    gt_720 = [
                        s
                        for s in streams
                        if _is_mp4(s)
                        and _height(s) > 720
                    ]

                    if not gt_720:
                        logger.error(
                            f"{vid}: no MP4 stream "
                            f"available at any resolution."
                        )

                        return None

                    min_h = min(
                        _height(s)
                        for s in gt_720
                    )

                    at_min_h = [
                        s
                        for s in gt_720
                        if _height(s) == min_h
                    ]

                    chosen = _pick_prefer_progressive(
                        at_min_h
                    )

                    selected_stream = chosen

                    selected_resolution = (
                        getattr(
                            chosen,
                            "resolution",
                            None,
                        )
                        or getattr(
                            chosen,
                            "res",
                            None,
                        )
                        or f"{min_h}p"
                    )

                    logger.debug(
                        f"{vid}: no ≤720p stream; "
                        f"picked lowest available >720p "
                        f"({selected_resolution})."
                    )

            logger.info(
                f"{vid}: download in "
                f"{selected_resolution} started "
                f"with pytubefix."
            )

            if os.path.exists(temp_video_file_path):
                os.remove(temp_video_file_path)

            selected_stream.download(
                temp_dir,
                filename=f"{vid}.mp4",
            )

            self.video_title = youtube_object.title

            os.replace(
                temp_video_file_path,
                final_video_file_path,
            )

            fps = self.get_video_fps(
                final_video_file_path
            )

            logger.info(
                f"{vid}: FPS={fps}."
            )

            return (
                final_video_file_path,
                vid,
                selected_resolution,
                fps,
            )

        except Exception as e:
            logger.error(
                f"{vid}: pytubefix download "
                f"method failed: {e}"
            )

            # ==============================================================
            # yt-dlp fallback
            # ==============================================================
            try:
                if (
                    self.update_package
                    and datetime.datetime.today().weekday() == 0
                ):
                    self.upgrade_package_if_needed(
                        "yt-dlp"
                    )

                youtube_url = (
                    f"https://www.youtube.com/watch?v={vid}"
                )

                yt_dlp_common_opts = {
                    "cookiesfrombrowser": ("brave",),
                    "remote_components": ["ejs:github"],
                    "noplaylist": True,
                }

                format_spec = (
                    "bv*[height<=720]+ba/"
                    "b[height<=720]/"
                    "bv*+ba/b"
                )

                selected_resolution = "best<=720p"

                ydl_opts = {
                    **yt_dlp_common_opts,
                    "outtmpl": os.path.join(
                        temp_dir,
                        f"{vid}.%(ext)s",
                    ),
                    "format": format_spec,
                    "merge_output_format": "mp4",
                    "final_ext": "mp4",
                    "verbose": False,
                }

                logger.info(
                    f"{vid}: download in "
                    f"{selected_resolution} started "
                    f"with yt-dlp."
                )

                if os.path.exists(
                    temp_video_file_path
                ):
                    os.remove(
                        temp_video_file_path
                    )

                with yt_dlp.YoutubeDL(
                    ydl_opts
                ) as ydl:
                    info = ydl.extract_info(
                        youtube_url,
                        download=True,
                    )

                self.video_title = (
                    info.get("title")
                    if isinstance(info, dict)
                    else None
                )

                os.replace(
                    temp_video_file_path,
                    final_video_file_path,
                )

                fps = self.get_video_fps(
                    final_video_file_path
                )

                logger.info(
                    f"{vid}: FPS={fps}."
                )

                return (
                    final_video_file_path,
                    vid,
                    selected_resolution,
                    fps,
                )

            except Exception as ytdlp_error:
                logger.error(
                    f"{vid}: yt-dlp download "
                    f"method failed: {ytdlp_error}"
                )

                return None

    def get_video_fps(self, video_file_path):
        """
        Retrieves the frames per second (FPS) of a video file using OpenCV.

        Parameters:
            video_file_path (str): Path to the video.

        Returns:
            int or None: Rounded FPS if successful.
        """
        try:
            video = cv2.VideoCapture(
                video_file_path
            )

            fps = video.get(
                cv2.CAP_PROP_FPS
            )

            video.release()

            return round(fps, 0)

        except Exception as e:
            logger.error(
                f"Failed to retrieve FPS: {e}"
            )

            return None
