DATA_ROOT = './data'
CHECKPOINTS_PATH = './checkpoints'

DATASET_TRAIN_FOLDER = "DIV2K_train_HR"
DATASET_TEST_FOLDER = "DIV2K_valid_HR"
PRINT_TIME = False

LOG_DIR = './logs'

TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 4
TEST_BATCH_SIZE = 1

CHANNELS = 3
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

LAMBDA_IMAGE_LOSS = 0.5
LAMBDA_SECRET_LOSS = 0.5

SAVE_FREQ = 1

NUM_BITS = 1024
DEVICE = 'mps'

# TEXT_EMBEDDING_MODULE = 'modules.text_embedding.LinearTextEmbedding1'
TEXT_EMBEDDING_MODULE = 'modules.text_embedding.LinearTextEmbedding1'
DWT_MODULE = 'modules.dwt.PRIS_DWT'
# IMAGE_EMBEDDING_MODULE = 'modules.image_embedding.WeightedImageEmbedding'
IMAGE_EMBEDDING_MODULE = 'modules.model.Hinet'
ATTACK_MODULE = 'modules.attack.NoneAttack'

LEARNING_RATE = 10 ** (-4.5)
WEIGHT_DECAY = 1e-5

