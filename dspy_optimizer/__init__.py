import logging

from dspy_optimizer.core.modules import (
    LinkedInStyleAnalyzer,
    LinkedInContentTransformer,
    LinkedInArticlePipeline,
    StylePipeline
)

from dspy_optimizer.core.optimizer import (
    configure_lm,
    optimize_analyzer,
    optimize_transformer,
    extract_optimized_prompts
)

from dspy_optimizer.utils.data import (
    load_examples,
    prepare_datasets
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)