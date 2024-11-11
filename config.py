DATA_ROOT = './data'
CHECKPOINTS_PATH = './checkpoints'
LOGS_PATH = './logs'

DATASET_TRAIN_FOLDER = "DIV2K_train_HR"
DATASET_TEST_FOLDER = "DIV2K_valid_HR"

LOG_DIR = './logs'

TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 2
TEST_BATCH_SIZE = 1
TRAIN_TEST_SPLIT = 0.8

CHANNELS = 3
ORIGINAL_IMAGE_HEIGHT = 224
ORIGINAL_IMAGE_WIDTH = 224
EMBEDDED_IMAGE_WEIGHT = 112
EMBEDDED_IMAGE_HEIGHT = 112

LAMBDA_IMAGE_LOSS = 0.5
LAMBDA_SECRET_LOSS = 0.5

SAVE_FREQ = 10

NUM_BITS = 112 * 112 / 2
DEVICE = 'mps'

TEXT_EMBEDDING_MODULE = 'modules.text_embedding.SimpleTextEmbedding'
# TEXT_EMBEDDING_MODULE = 'modules.text_embedding.MiddleQuarterSquareTextEmbedding'
TEXT_EMBEDDING_CHANNEL = 1
# TEXT_EMBEDDING_MODULE = 'modules.text_embedding.VitTextEmbedding'
DWT_MODULE = 'modules.dwt.PRIS_DWT'
IMAGE_EMBEDDING_MODULE = 'modules.model.Hinet'

ATTACK_MODULE = 'modules.attack.NoneAttack'
# ATTACK_MODULE = 'modules.attack.JPEGCompressionPRISAttack'
# ATTACK_MODULE = 'modules.attack.OcclusionAttack'
# ATTACK_MODULE = 'modules.attack.OrderOcclusionAttack'
LEARNING_RATE = 10 ** (-4)

# DISCRIMINATOR_MODULE = 'modules.discriminator.Discriminator_AvgPool'
DISCRIMINATOR_MODULE = 'modules.discriminator.ViTDiscriminator'
DISCRIMINATOR_LEARNING_RATE = 10 ** (-4)
DISCRIMINATOR_INPUT_CHANNELS = 1

WEIGHT_DECAY = 1e-5
