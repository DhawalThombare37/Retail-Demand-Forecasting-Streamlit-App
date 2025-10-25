import logging

def setup_logger(name="retail_forecast", log_file="logs/app.log", level=logging.INFO):
    """
    Create and configure a logger.
    """
    logger = logging.getLogger(name)
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger
