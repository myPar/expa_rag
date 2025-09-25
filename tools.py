
import magic


def is_text(file_path: str):
    try:
        return magic.from_file(file_path, mime=True).split('/')[0] == 'text'
    except Exception:
        # issue only on cyrillic file names on windows
        return file_path.split(".")[-1] in ['txt', 'md']