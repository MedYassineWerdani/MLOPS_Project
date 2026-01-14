from zenml import pipeline, step
from zenml.logger import get_logger
import logging
import subprocess

logger = get_logger(__name__)


@step
def train_step() -> str:
    logger.info("Running train_step (executes train.py)")
    # Run training as a separate process to avoid import-time side effects
    subprocess.run(["python", "train.py"], check=True)
    return "checkpoints"


@pipeline
def training_pipeline():
    train_step()


def main():
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting ZenML pipeline for training")
    pipeline_run = training_pipeline()
    pipeline_run.run()
    logger.info("Pipeline run finished")


if __name__ == "__main__":
    main()
