import sys
from pathlib import Path

class TeeLogger:
    def __init__(self, log_file: Path):
        self.log_file = open(log_file, "w", encoding="utf-8")
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, message):
        self.stdout.write(message)
        self.log_file.write(message)

    def flush(self):
        self.stdout.flush()
        self.log_file.flush()

    def close(self):
        sys.stdout = self.stdout
        self.log_file.close()


class TeeLoggerContext:
    def __init__(self, log_file: Path):
        self.logger = TeeLogger(log_file)

    def __enter__(self):
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.close()