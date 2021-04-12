
import tensorflow as tf
from flask import Flask, request
import numpy as np

DEFAULT_MODEL_PATH = './unet_attn_BCE_aug_recomp_20210411-235414.h5'

app = Flask(__name__)

@app.route('/inference', methods=["POST"])
def inference():
    data = request.json
    img_arr = np.array(data["image"], dtype=np.uint8)
    
    # load and instantiate model
    model = tf.keras.models.load_model(DEFAULT_MODEL_PATH)
    
    # Get them predictions
    mask_prediction = model.predict(np.expand_dims(x, axis=0))[0] > 0.5 # 
    mask_prediction = mask_prediction[:,:,0] # get mask layer
    return {"prediction": mask_prediction.tolist()}

if __name__ == '__main__':
    app.run(debug=True)

