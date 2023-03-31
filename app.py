from flask import Flask, request, jsonify
import pickle
import numpy as np
# from flask_socketio import SocketIO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
#socketio = SocketIO(app, cors_allowed_origins='*')

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, len(to_predict_list))
    loaded_model = pickle.load(open("pkl/heart_disease_SVC.pkl", "rb"))
    print(loaded_model)
    result = loaded_model.predict_proba(to_predict)
    return round(result[0][1]*100, 2)

@app.route('/api', methods = ['GET'])
def returnProb():
    # d = {}
    # inputchr = str(request.args['query'])
    # answer = str(ord(inputchr))
    # d['output'] = answer
    # return d
    d = {}
    #X = ['age', 'sex', 'cp', 'trtbps', 'chol',
    #     'fbs', 'restecg', 'thalachh', 'exng',
    #     'oldpeak', 'slp', 'caa', 'thall']
    to_predict_list = []
    #for x in X:
    #    to_predict_list.append(int(request.args[x]))
    # print(to_predict_list)
    to_predict_list.append(float(request.args['BMI'])) 
    to_predict_list.append(int(request.args['Smoking']))
    to_predict_list.append(int(request.args['AlcoholDrinking']))
    to_predict_list.append(int(request.args['Stroke']))
    to_predict_list.append(int(request.args['PhysicalHealth']))
    to_predict_list.append(int(request.args['MentalHealth']))
    to_predict_list.append(int(request.args['DiffWalking']))
    to_predict_list.append(int(request.args['sex']))
    to_predict_list.append(int(request.args['AgeCategory']))
    to_predict_list.append(int(request.args['Diabetic']))
    to_predict_list.append(int(request.args['PhysicalActivity']))
    to_predict_list.append(int(request.args['GenHealth']))
    to_predict_list.append(int(request.args['SleepTime']))
    to_predict_list.append(int(request.args['Asthma']))
    to_predict_list.append(int(request.args['KidneyDisease']))
    to_predict_list.append(int(request.args['SkinCancer']))

    d['output'] = str(ValuePredictor(to_predict_list))
    # return str(to_predict_list)
    return d

if __name__ =="__main__":
    app.run(debug=True)

# api?age=30&sex=0&cp=1&trtbps=1&chol=1&fbs=1&restecg=1&thalachh=1&exng=1&oldpeak=1&slp=1&caa=1&thall=1
