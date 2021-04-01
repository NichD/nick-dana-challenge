
import tensorflow as tf
from flask import Flask, request
import numpy as np

app = Flask(__name__)

@app.route('/inference', methods=["POST"])
def inference():
    data = request.json
    img_arr = np.array(data["image"], dtype=np.uint8)
    # TODO: Replace with a call to your model
    random_mask = (np.random.uniform(size=(img_arr.shape[:2])) > 0.5).astype(np.uint8)
    return {"prediction": random_mask.tolist()}

if __name__ == '__main__':
    app.run(debug=True)

