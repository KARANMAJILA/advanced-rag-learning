
from pathlib import Path
import re


class DocumentLoader:

    def __init__(self, directory: str):
        self.directory = Path(directory)

        if not self.directory.exists():
            raise FileNotFoundError(
                f"Directory not found: {directory}"
            )

    def clean_text(self, text: str) -> str:
        """Normalize whitespace in document text."""

        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def load_txt(self):
        """Load all TXT files from the directory."""

        documents = []

        for file_path in self.directory.glob("*.txt"):

            try:
                text = file_path.read_text(
                    encoding="utf-8"
                )

                text = self.clean_text(text)

                if not text:
                    continue

                documents.append({
                    "document_id": file_path.stem,
                    "filename": file_path.name,
                    "text": text,
                    "source": str(file_path)
                })

            except (OSError, UnicodeDecodeError) as error:
                print(f"Could not load {file_path}: {error}")

        return documents
