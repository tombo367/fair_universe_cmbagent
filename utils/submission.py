import json
import zipfile
from pathlib import Path


def save_json_zip(submission_dir, json_file_name, zip_file_name, data):
    """Write `data` to JSON and zip it on its own, as the challenge expects."""
    submission_dir = Path(submission_dir)
    submission_dir.mkdir(parents=True, exist_ok=True)
    json_path = submission_dir / json_file_name
    zip_path = submission_dir / zip_file_name

    json_path.write_text(json.dumps(data))
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname=json_file_name)
    json_path.unlink()
    return str(zip_path)
