from flask import Flask, jsonify
from api.predict import predict_route

app = Flask(__name__)
app.register_blueprint(predict_route, url_prefix='/api')

if __name__ == '__main__':
    app.run()