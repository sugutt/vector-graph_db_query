import os
from config import IGNORE_DIRS, ALLOWED_EXTS

def scan_project(root_path):
    code_files = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        for filename in filenames:
            if os.path.splitext(filename)[1] in ALLOWED_EXTS:
                code_files.append(os.path.join(dirpath, filename))
    return code_files
