import os


def main() -> None:
    """Entry point for the application."""
    app_name = os.getenv("APP_NAME", "Lang Teacher Agent")
    app_env = os.getenv("APP_ENV", "development")
    print(f"{app_name} is running in {app_env} mode.")


if __name__ == "__main__":
    main()
