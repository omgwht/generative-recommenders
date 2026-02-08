# pyre-unsafe

from generative_recommenders.research.data.preprocessor import get_common_preprocessors


def main() -> None:
    get_common_preprocessors()["epinions"].preprocess_rating()


if __name__ == "__main__":
    main()
