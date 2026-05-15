"""Entry point so ``python -m visionservex`` works."""

from visionservex.cli.main import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
