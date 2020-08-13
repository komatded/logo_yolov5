import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s - %(name)s')
logger = logging.getLogger()

IMAGE_SIZE = 608
MODEL_PATH = 'resources/model.pt'
# Веса модели лежат на ЯндексДиске: https://yadi.sk/d/NRbqEmWp2LN8vg
