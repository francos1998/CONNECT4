import anvil.server
import numpy as np
import tensorflow as tf
import os


# ----------------------------
# Custom Layers (required for transformer model)
# ----------------------------
class PositionalIndex(tf.keras.layers.Layer):
    """Generate positional indices for embeddings"""
    def call(self, x):
        bs = tf.shape(x)[0]  # batch size
        number_of_vectors = tf.shape(x)[1]  # number of patches (42 for Connect 4)
        indices = tf.range(number_of_vectors)  # 0, 1, 2, ..., 41
        indices = tf.expand_dims(indices, 0)  # shape: (1, 42)
        return tf.tile(indices, [bs, 1])  # repeat for each batch: (batch, 42)

class ClassTokenIndex(tf.keras.layers.Layer):
    """Generate index for class token"""
    def call(self, x):
        bs = tf.shape(x)[0]  # batch size
        number_of_vectors = 1  # we only want 1 class token
        indices = tf.range(number_of_vectors)  # just [0]
        indices = tf.expand_dims(indices, 0)
        return tf.tile(indices, [bs, 1])  # (batch, 1)

# ----------------------------
# Anvil Server Uplink
# ----------------------------
ANVIL_UPLINK_KEY = os.environ["ANVIL_UPLINK_KEY"]
anvil.server.connect(ANVIL_UPLINK_KEY)

# ----------------------------
# Model Paths (inside Docker container)
# ----------------------------
CNN_MODEL_PATH = '/home/bitnami/connect-four/models/cnnmodel2.h5'
TRANSFORMER_MODEL_PATH = '/home/bitnami/connect-four/models/transformer2.keras'

# ----------------------------
# Load Models
# ----------------------------
print("Loading CNN model...")
cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
print("CNN model loaded and ready.")

print("Loading Transformer model...")
transformer_model = tf.keras.models.load_model(
    TRANSFORMER_MODEL_PATH,
    custom_objects={
        'PositionalIndex': PositionalIndex,
        'ClassTokenIndex': ClassTokenIndex
    }
)
print("Transformer model loaded and ready.")

# ----------------------------
# Helper Functions
# ----------------------------
def board_to_cnn_input(board, player):
    """
    Convert flat board list to 6x7x2 numpy array for CNN.
    Channel 0 = AI's pieces, Channel 1 = opponent's pieces
    """
    board_array = np.array(board, dtype=np.float32).reshape(6, 7)
    opponent = 2 if player == 1 else 1
    channel_ai = (board_array == player).astype(np.float32)
    channel_opp = (board_array == opponent).astype(np.float32)
    return np.stack([channel_ai, channel_opp], axis=-1).reshape(1, 6, 7, 2)

def board_to_transformer_input(board, player):
    """
    Convert flat board list to 42x1 numpy array for Transformer.
    Encode: empty=0, AI=1, opponent=-1
    """
    board_array = np.array(board, dtype=np.float32).reshape(6, 7)
    opponent = 2 if player == 1 else 1
    
    # Remap: AI pieces -> 1, opponent pieces -> -1, empty -> 0
    encoded = np.zeros_like(board_array)
    encoded[board_array == player] = 1.0
    encoded[board_array == opponent] = -1.0
    
    # Flatten to 42x1
    return encoded.reshape(1, 42, 1)

def get_best_move(model, board, player, model_type='cnn'):
    """
    Run prediction and return the best valid column.
    model_type: 'cnn' or 'transformer'
    """
    if model_type == 'cnn':
        input_tensor = board_to_cnn_input(board, player)
    else:  # transformer
        input_tensor = board_to_transformer_input(board, player)
    
    predictions = model.predict(input_tensor, verbose=0)[0]
    sorted_cols = np.argsort(-predictions)  # highest probability first
    board_array = np.array(board).reshape(6, 7)
    
    for col in sorted_cols:
        if board_array[0][col] == 0:  # column not full
            return int(col)
    return -1  # board full

# ----------------------------
# Anvil Callable Functions
# ----------------------------
@anvil.server.callable
def cnn_best_move(board, player):
    """
    board: flat list of 42 ints (0=empty, 1=red, 2=yellow), row by row
    player: int, which player the AI is (1 or 2)
    returns: int, column (0-6) to play
    """
    print(f'CNN predict called. Player: {player}')
    col = get_best_move(cnn_model, board, player, model_type='cnn')
    print(f'CNN chose column: {col}')
    return col

@anvil.server.callable
def transformer_best_move(board, player):
    """
    board: flat list of 42 ints (0=empty, 1=red, 2=yellow), row by row
    player: int, which player the AI is (1 or 2)
    returns: int, column (0-6) to play
    """
    print(f'Transformer predict called. Player: {player}')
    col = get_best_move(transformer_model, board, player, model_type='transformer')
    print(f'Transformer chose column: {col}')
    return col

# ----------------------------
# Run Server
# ----------------------------
print("Connect-4 AI server running and waiting for Anvil calls...")
anvil.server.wait_forever()
