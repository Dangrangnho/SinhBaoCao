"""Small example application used by the end-to-end pipeline test."""

from database import initialize_database, save_user


def main() -> None:
    """Initialize the database and insert one sample user."""
    initialize_database()
    save_user("demo@example.com")
    print("Demo user saved")


if __name__ == "__main__":
    main()
