import logging
import os

from dspy_optimizer.utils.data import (
    load_examples,
    prepare_datasets
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

for logger_name in ['litellm', 'httpx', 'numexpr', 'datasets', 'tqdm']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

os.environ['TQDM_DISABLE'] = '1'