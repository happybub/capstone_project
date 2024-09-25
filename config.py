DATA_ROOT = './data'
CHECKPOINTS_PATH = './checkpoints'
DATASET_TRAIN_FOLDER = "DIV2K_train_HR"
DATASET_TEST_FOLDER = "DIV2K_valid_HR"
TRAIN_BATCH_SIZE = 1
VAL_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1

CHANNELS = 3
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

NUM_BITS = 7
DEVICE = 'cpu'

TEXT_EMBEDDING_MODULE = 'modules.text_embedding.TextEmbeddingModule'
DWT_MODULE = 'modules.dwt.PRIS_DWT'
IMAGE_EMBEDDING_MODULE = 'modules.image_embedding.WeightedImageEmbedding'
ATTACK_MODULE = 'modules.attack.GaussianNoiseAttack'

LEARNING_RATE = 1e-5

