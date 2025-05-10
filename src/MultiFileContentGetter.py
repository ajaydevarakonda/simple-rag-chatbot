import os
import asyncio

import aiofiles

from src.ContentGetter import ContentGetter

class MultiFileContentGetter(ContentGetter):
    @staticmethod
    def read_multiple_files_sync(file_paths):
        f_strs = []

        for f_path in file_paths:
            with open(f_path, 'r') as f:
                f_strs.append(f.read())

        return f_strs

    def get_content(self, file_paths=[]):
        results = self.read_multiple_files_sync(file_paths)
        result = '\n'.join(results)
        return result