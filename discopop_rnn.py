import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
import json
import re

FUNCTION_EPOCHS = 20
LOG_FILE = 'chat_log.txt'

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

client = genai.Client(api_key=GEMINI_API_KEY)
sys_msg = """
You are a machine learning researcher focused on discovering novel loss functions to improve stock price prediction. Your models (RNN, LSTM, GRU) are already compiled in TensorFlow and trained using sequences of historical stock prices.

When you respond, output a JSON where:

- "thought" is your reasoning behind the design of the new loss function (you can reference known time-series or regression loss designs from literature like quantile loss, Huber loss, etc.).

- "name" is a short descriptive name for your new loss function.

- "code" is the exact TensorFlow-compatible Python function implementing this loss.

The loss function must follow this interface:
```python
def custom_loss(y_true, y_pred):
    # your code here
    return loss
```

You can define constants (like thresholds) inside the function, but do not rely on external variables or classes. Be creative and use your knowledge of time-series forecasting, volatility modeling, and robust regression.

You are deeply familiar with:

- Traditional losses like MSE, MAE, Huber.

- Financial modeling considerations like penalizing volatility misprediction, tail risks, or direction accuracy.

- Research from probabilistic forecasting, heteroscedastic regression, and robust statistics.

Return only a JSON like the following:
```json
{
  "thought": "I want to penalize the model more when it predicts the wrong direction of price movement even if the absolute error is small. I’ll combine MSE with a directional penalty term.",
  "name": "directional_mse",
  "code": "def custom_loss(y_true, y_pred):\n    import tensorflow as tf\n    direction_true = tf.sign(y_true[:, 1:] - y_true[:, :-1])\n    direction_pred = tf.sign(y_pred[:, 1:] - y_pred[:, :-1])\n    direction_penalty = tf.reduce_mean(tf.cast(tf.not_equal(direction_true, direction_pred), tf.float32))\n    mse = tf.reduce_mean(tf.square(y_true - y_pred))\n    return mse + 0.5 * direction_penalty"
}
```

Your goal is to discover creative, practical loss functions that lead to improved generalization and robustness in stock price prediction models.
"""

def escape_code_block(json_like_text):
    # Find the code block and escape its inner newlines and quotes properly
    match = re.search(r'"code":\s*"((?:.|\n)*?)"', json_like_text)
    if match:
        code_block = match.group(1)
        # Escape internal double quotes and newlines
        escaped_code = code_block.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
        return json_like_text.replace(code_block, escaped_code)
    return json_like_text

if __name__ == '__main__':
    chat_log = []
    prev_log = ''
    try:
        with open(LOG_FILE, 'r') as f:
            prev_log = f.read()
    except FileNotFoundError:
        print('Log not found, creating a file')

    log_file = open(LOG_FILE, 'a')
    try:
        chat = client.chats.create(model='gemini-2.0-flash', config=types.GenerateContentConfig(
            system_instruction=sys_msg,
        ))
        response = chat.send_message('Please generate the next one')
        if response.text is not None:
            uncode_json = '\n'.join(response.text.split('\n')[1:-1])
            parsed_data = json.loads(escape_code_block(uncode_json))
            print(parsed_data['code'])
    except KeyboardInterrupt:
        print('Keyboard Interrupt caught')
    finally:
        log_file.close()
