import os
import json
import io
import logging
import redis
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import tensorflow as tf

# ═══════════════════════════════════════════════
#  Logging
# ═══════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  [%(levelname)s]  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════
#  Flask Setup
# ═══════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates')
)

CORS(app)

# ═══════════════════════════════════════════════
#  Redis Connection
# ═══════════════════════════════════════════════

def get_redis_client():
    """
    Returns a Redis client.
    Raises a clear RuntimeError if the connection fails.
    """
    try:
        client = redis.Redis(
            host=os.getenv('REDIS_HOST', '127.0.0.1'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            decode_responses=True,
            socket_connect_timeout=3
        )
        client.ping()
        return client
    except redis.exceptions.ConnectionError as e:
        raise RuntimeError(f"Redis connection failed: {e}")

try:
    r = get_redis_client()
    log.info("Redis connected successfully.")
except RuntimeError as e:
    log.warning(e)
    r = None

# ═══════════════════════════════════════════════
#  Labels
# ═══════════════════════════════════════════════

LABELS = [
    'butter_naan',    'pav_bhaji',      'Sandwich',
    'chicken_curry',  'Hot Dog',         'cheesecake',
    'sushi',          'chai',            'burger',
    'ice_cream',      'kadai_paneer',    'Baked Potato',
    'chapati',        'masala_dosa',     'dal_makhani',
    'Donut',          'jalebi',          'fried_rice',
    'chole_bhature',  'kulfi',           'kaathi_rolls',
    'dhokla',         'Fries',           'omelette',
    'pakode',         'momos',           'paani_puri',
    'samosa',         'Taco',            'idli',
    'Taquito',        'Crispy Chicken',  'pizza',
    'apple_pie'
]

NUM_CLASSES = len(LABELS)

# ═══════════════════════════════════════════════
#  Model Input Sizes
# ═══════════════════════════════════════════════

MODEL_INPUT_SIZES = {
    'custom_cnn': (256, 256),
    'vgg16':      (224, 224),
    'resnet50':   (224, 224),
}

WEIGHT_FILES = {
    'custom_cnn': 'food_classification_weights.h5',
    'vgg16':      'vgg16_food_classification_weights.h5',
    'resnet50':   'resnet50_food_classification_weights.h5',
}

# ═══════════════════════════════════════════════
#  Model Cache
# ═══════════════════════════════════════════════

_model_cache: dict = {}

# ═══════════════════════════════════════════════
#  Model Builders
# ═══════════════════════════════════════════════

def build_custom_cnn(
    input_shape: tuple = (256, 256, 3),
    num_classes: int = NUM_CLASSES
) -> tf.keras.Model:
    """5-block Custom CNN with increasing filter depth."""

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    for filters in [32, 64, 128, 256, 512]:
        x = tf.keras.layers.Conv2D(
            filters, (3, 3),
            activation='relu',
            padding='same'
        )(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Flatten()(x)

    for units in [1024, 512, 256, 128, 64]:
        x = tf.keras.layers.Dense(units, activation='relu')(x)

    outputs = tf.keras.layers.Dense(
        num_classes, activation='softmax'
    )(x)

    return tf.keras.Model(inputs, outputs, name='custom_cnn')


def build_vgg16(num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    """VGG-16 backbone (no pretrained weights) + custom head."""

    base = tf.keras.applications.VGG16(
        weights=None,
        include_top=False,
        input_shape=(224, 224, 3)
    )
    x = tf.keras.layers.Flatten()(base.output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation='softmax'
    )(x)

    return tf.keras.Model(base.input, outputs, name='vgg16')


def build_resnet50(num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    """ResNet-50 backbone (no pretrained weights) + custom head."""

    base = tf.keras.applications.ResNet50(
        weights=None,
        include_top=False,
        input_shape=(224, 224, 3)
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation='softmax'
    )(x)

    return tf.keras.Model(base.input, outputs, name='resnet50')


_BUILDERS = {
    'custom_cnn': build_custom_cnn,
    'vgg16':      build_vgg16,
    'resnet50':   build_resnet50,
}

# ═══════════════════════════════════════════════
#  Model Loader
# ═══════════════════════════════════════════════

def load_model(model_name: str) -> tf.keras.Model:
    """
    Loads and caches a Keras model by name.
    Raises ValueError for unknown names, FileNotFoundError if weights missing.
    """
    if model_name in _model_cache:
        return _model_cache[model_name]

    if model_name not in _BUILDERS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(_BUILDERS.keys())}"
        )

    weight_path = os.path.join(BASE_DIR, WEIGHT_FILES[model_name])

    if not os.path.isfile(weight_path):
        raise FileNotFoundError(
            f"Weight file not found: {weight_path}"
        )

    log.info(f"Building model: {model_name}")
    model = _BUILDERS[model_name]()

    log.info(f"Loading weights from: {weight_path}")
    try:
        model.load_weights(weight_path)
    except Exception:
        log.warning(
            f"Strict weight load failed for {model_name}. "
            "Retrying with skip_mismatch=True."
        )
        model.load_weights(
            weight_path,
            skip_mismatch=True,
            by_name=True
        )

    _model_cache[model_name] = model
    log.info(f"Model '{model_name}' loaded and cached.")

    return model

# ═══════════════════════════════════════════════
#  Image Preprocessing
# ═══════════════════════════════════════════════

def preprocess_image(
    image_bytes: bytes,
    size: tuple,
    model_name: str
) -> np.ndarray:
    """
    Decodes raw image bytes → resizes → normalises
    according to the target model's expected input format.
    Returns a (1, H, W, 3) float32 batch array.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(size, Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)

    if model_name == 'vgg16':
        from tensorflow.keras.applications.vgg16 import preprocess_input
        return preprocess_input(img_batch)

    if model_name == 'resnet50':
        from tensorflow.keras.applications.resnet50 import preprocess_input
        return preprocess_input(img_batch)

    # Custom CNN — simple [0, 1] normalisation
    return img_batch / 255.0

# ═══════════════════════════════════════════════
#  Per-Prediction Metrics
#
#  All metrics are derived from the softmax
#  probability vector alone (no ground-truth needed).
#
#  Confidence  = P(top class)
#  Precision   = P(top) / Σ P(classes above mean)
#                ≈ exclusivity of the prediction
#  Recall      = P(top) / (P(top) + mean(rest))
#                ≈ signal-to-noise ratio
#  F1-Score    = harmonic mean of Precision & Recall
# ═══════════════════════════════════════════════

def compute_metrics(probs: np.ndarray) -> dict:
    """
    probs : 1-D float array of softmax probabilities
    Returns confidence, precision, recall, f1_score (0–100 %).
    """
    probs    = np.array(probs, dtype=np.float64)
    top_idx  = int(np.argmax(probs))
    p_top    = float(probs[top_idx])

    # Confidence
    confidence = round(p_top * 100, 2)

    # Precision — share of "candidate" mass taken by the top class
    threshold      = float(np.mean(probs))
    candidate_mass = float(np.sum(probs[probs > threshold]))
    precision      = p_top / candidate_mass if candidate_mass > 0 else p_top
    precision      = round(min(precision, 1.0) * 100, 2)

    # Recall — top-class signal vs background noise
    rest_mean = float(np.mean(np.delete(probs, top_idx)))
    denom     = p_top + rest_mean
    recall    = (p_top / denom) if denom > 0 else 1.0
    recall    = round(min(recall, 1.0) * 100, 2)

    # F1-Score
    p_f = precision / 100.0
    r_f = recall   / 100.0
    f1  = (2 * p_f * r_f / (p_f + r_f)) if (p_f + r_f) > 0 else 0.0
    f1_score = round(f1 * 100, 2)

    return {
        'confidence': confidence,
        'precision':  precision,
        'recall':     recall,
        'f1_score':   f1_score,
    }

# ═══════════════════════════════════════════════
#  Nutrition Lookup (Redis)
# ═══════════════════════════════════════════════

def get_nutrition(food_name: str) -> dict | None:
    """
    Fetches per-100 g nutrition from the Redis
    'food_details' key (JSON object).
    Returns None if Redis is unavailable or food not found.
    """
    if r is None:
        log.warning("Redis unavailable — skipping nutrition lookup.")
        return None

    try:
        raw = r.get('food_details')
        if not raw:
            log.warning("'food_details' key not found in Redis.")
            return None

        food_details = json.loads(raw)
        nutrition    = food_details.get(food_name)

        if not nutrition:
            log.info(f"No nutrition data for '{food_name}' in Redis.")
            return None

        return {
            'calories': nutrition.get('calories', 0),
            'protein':  nutrition.get('protein',  0),
            'carbs':    nutrition.get('carbs',    0),
            'fats':     nutrition.get('fats',     0),
            'fiber':    nutrition.get('fiber',    0),
        }

    except json.JSONDecodeError as e:
        log.error(f"Redis JSON parse error: {e}")
    except Exception as e:
        log.error(f"Redis lookup error: {e}")

    return None

# ═══════════════════════════════════════════════
#  Routes
# ═══════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects multipart/form-data with:
      - image  : image file (JPG / PNG / WEBP)
      - model  : one of custom_cnn | vgg16 | resnet50
    Returns JSON with prediction, metrics, top-5, and nutrition.
    """
    # ── Validate image ──────────────────────────
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({'error': 'Empty filename — please select a file.'}), 400

    # ── Validate model name ─────────────────────
    model_name = (
        request.form.get('model', 'custom_cnn')
        .strip().lower().replace(' ', '_')
    )

    if model_name not in MODEL_INPUT_SIZES:
        return jsonify({
            'error': f"Invalid model '{model_name}'. "
                     f"Choose from: {list(MODEL_INPUT_SIZES.keys())}"
        }), 400

    try:
        image_bytes = image_file.read()
        size        = MODEL_INPUT_SIZES[model_name]

        # Load & run inference
        model          = load_model(model_name)
        processed      = preprocess_image(image_bytes, size, model_name)
        preds          = model.predict(processed, verbose=0)[0]

        # Primary prediction
        top_idx    = int(np.argmax(preds))
        label      = LABELS[top_idx]
        confidence = round(float(preds[top_idx]) * 100, 2)

        log.info(
            f"Prediction → {label} ({confidence}%) "
            f"| model={model_name}"
        )

        # Metrics
        metrics = compute_metrics(preds)

        # Top-5
        top5 = [
            {
                'label':      LABELS[i],
                'confidence': round(float(preds[i]) * 100, 2)
            }
            for i in np.argsort(preds)[::-1][:5]
        ]

        # Nutrition
        nutrition = get_nutrition(label)

        return jsonify({
            'predicted_label': label,
            'confidence':      confidence,
            'model_used':      model_name,
            'top5':            top5,
            'nutrition':       nutrition,
            'metrics':         metrics,
        })

    except (ValueError, FileNotFoundError) as e:
        log.error(f"Setup error: {e}")
        return jsonify({'error': str(e)}), 400

    except Exception as e:
        log.exception(f"Unexpected prediction error: {e}")
        return jsonify({'error': 'Internal server error. Check server logs.'}), 500


# ── Health check (useful for container probes) ──
@app.route('/health')
def health():
    redis_ok = False
    if r is not None:
        try:
            r.ping()
            redis_ok = True
        except Exception:
            pass

    return jsonify({
        'status':       'ok',
        'redis':        'connected' if redis_ok else 'unavailable',
        'models_loaded': list(_model_cache.keys()),
        'num_classes':   NUM_CLASSES,
    })

# ═══════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════

if __name__ == '__main__':
    app.run(
        debug=os.getenv('FLASK_DEBUG', 'true').lower() == 'true',
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000))
    )