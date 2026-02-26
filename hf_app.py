"""Hugging Face Spaces entrypoint for the API-based ChatPDF app."""

from methods.api.api_app import page


if __name__ == "__main__":
    page()
