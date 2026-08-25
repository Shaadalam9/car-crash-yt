# Car Crash YouTube Segmentation Pipeline

This project discovers car crash videos and compilations on YouTube, rejects
unsuitable metadata, downloads accepted videos, separates compilations into
complete source clips, identifies clips containing real crashes or near
collisions, extracts structured visible information, and writes resumable JSON
and CSV research outputs.

It is derived from the architecture of
[`walking-yt`](https://github.com/Shaadalam9/walking-yt), with crash specific
prompts, full clip boundary verification, unrestricted video duration, and no
minimum or maximum accepted segment duration.

## What “full segment” means

A segment is the complete source clip between two verified compilation edits.
The segment starts when that source recording begins and ends at the next real
edit, rather than starting shortly before impact or stopping immediately after
it. If an upload contains no edit, the complete upload is one segment.

Segments and uploads may have any positive duration. The visual model still
receives a bounded number of uniformly distributed frames for predictable GPU
memory use, but the stored start and end times always describe the complete
source clip.

## Pipeline

1. YouTube discovery searches configured crash and dashcam queries.
2. A Qwen text model rejects games, simulations, films, crash tests,
   motorsport-only videos, and unrelated content before download.
3. `yt-dlp` downloads accepted videos into a temporary directory and moves only
   validated media into the processing directory.
4. FFmpeg proposes scene transitions using standard and adaptive detectors.
5. NVIDIA Cosmos 3 Nano verifies whether every proposal is a real compilation
   edit rather than camera motion, impact shake, occlusion, or exposure change.
6. Verified edit points define complete, variable length source clips.
7. Cosmos reviews a temporally uniform sample covering each complete clip.
8. Accepted crash segments are optionally geocoded only when metadata or
   readable embedded text explicitly supports a location.
9. State JSON and one-row-per-segment CSV output are updated atomically.

## Extracted information

Every accepted segment stores:

- YouTube video ID and stable segment ID
- full segment start, impact and end times
- crash confidence and short factual description
- crash type and camera viewpoint
- visible road user count and road user types
- road environment, time of day, weather and road condition
- visible outcomes
- author supplied timestamp and chapter labels
- embedded location text and its evidence source
- locality, state, country, ISO3, continent and coordinates when supported
- model version and raw model response in the JSON state

The prompt explicitly prohibits inferring fault, injury, fatality, intent,
identity or legal responsibility.

## Requirements

- Python 3.12.13
- `uv`
- FFmpeg and FFprobe
- `yt-dlp`
- a CUDA GPU suitable for Cosmos 3 Nano
- YouTube Data API credentials
- YouTube cookies when required by YouTube

Install the environment:

```bash
uv sync
```

Generate and commit a lock file for a frozen deployment when network access is
available:

```bash
uv lock
uv sync
```

Check the external commands:

```bash
ffmpeg -version
ffprobe -version
yt-dlp --version
ffmpeg -hide_banner -hwaccels
```

## Configuration

Create the active configuration and secret files:

```bash
cp default.config config
cp default.secret secret
```

Add one or more YouTube Data API keys to `secret`:

```json
{
  "YOUTUBE_API_KEYS": ["your-key"]
}
```

`config`, `secret`, `cookies.txt`, downloads, state and outputs are ignored by
Git.

To process a particular video before normal search discovery, put its eleven
character ID in `SEED_VIDEO_IDS`. For example:

```json
{
  "SEED_VIDEO_IDS": ["jJXT2zGlSc0"]
}
```

This value is only the ID, not the complete URL. Normal discovery continues in
the same cycle until `MAX_NEW_CANDIDATES` is reached.

Important settings include:

| Setting | Purpose |
|---|---|
| `DISCOVERY_QUERIES` | YouTube search terms |
| `SEED_VIDEO_IDS` | Explicit video IDs to add before search |
| `MAX_NEW_CANDIDATES` | Maximum new videos in a discovery batch |
| `MAX_VIDEOS_PER_RUN` | Maximum downloaded videos processed in one cycle |
| `CONTINUOUS_MODE` | Repeat discovery and processing cycles |
| `MIN_TEXT_CONFIDENCE` | Metadata acceptance threshold |
| `MIN_BOUNDARY_CONFIDENCE` | Verified compilation edit threshold |
| `MIN_CRASH_CONFIDENCE` | Accepted segment threshold |
| `SAMPLE_FRAME_COUNT` | Approximate frame budget across a complete segment |
| `SAMPLE_MAX_FPS` | Maximum sampling rate for short clips |
| `CUT_BACKEND` | `ffmpeg_cuda`, `ffmpeg_cpu`, or `auto` |
| `SCENE_THRESHOLD` | Standard FFmpeg scene sensitivity |
| `SCDET_THRESHOLD` | Adaptive scene detector sensitivity |
| `DELETE_VIDEO_AFTER_PROCESSING` | Delete a download after its final decision |
| `ENABLE_GEOCODING` | Resolve explicit locations with Nominatim |

There is deliberately no minimum video duration, maximum video duration,
minimum segment duration, or maximum segment duration setting.

## Running

Run once:

```bash
python main.py
```

or:

```bash
python -m car_crash_pipeline
```

Set `CONTINUOUS_MODE` to `true` for repeated batches. Stop with `Ctrl+C`; the
current state and CSV are saved and the next run resumes unfinished work.

## Outputs

By default, files are written below `data/`:

- `state.json`: authoritative detailed state, model responses, boundary reviews
  and accepted or rejected segment decisions
- `crash_segments.csv`: one row for every accepted full crash source clip
- `geocode_cache.json`: reusable Nominatim responses
- `videos/`: validated downloads, retained only when configured

The CSV is a convenient analysis table. The JSON state remains the richer audit
record and should be retained for reproducibility.

## Validation

The fast tests do not load a model or access YouTube:

```bash
python -m unittest discover -s tests -v
```

They cover unrestricted segment construction, cut proposal merging, crash
decision normalisation, strict location evidence, impact timing and CSV row
construction.

## Research and platform responsibilities

Use the pipeline only where collection and processing comply with YouTube's
terms, applicable copyright rules, research ethics approval, privacy
requirements and local law. Crash footage may contain distressing or sensitive
material. Store it securely, minimise retention, and avoid publishing personally
identifiable frames or unsupported conclusions.
