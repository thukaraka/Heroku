import urllib
import json
import os
import  pandas as pd

from flask import Flask
from flask import request
from flask import make_response
import pickle

# Flask app should start in global layout
app = Flask(__name__)
model=pickle.load(open('model.pkl','rb')) 
 
@app.route('/webhook', methods=['POST'])
def webhook():
        req = request.get_json(silent=True, force=True)
        res = json.dumps(req['queryResult']['parameters'], indent=4)
        int_features=json.loads(res)
        print(int_features)
        final_features=pd.DataFrame({'age':[int(int_features['age']['amount'])],'sex':[int(int_features['sex'])],'cp':[int(int_features['cp'])],'trestbps':[int(int_features['trestbps'])],'chol':[int(int_features['chol'])],'fbs':[int(int_features['fbs'])],'restecg':[int(int_features['restecg'])],'thalach':[int(int_features['thalach'])],'exang':[int(int_features['exang'])],'oldpeak':[float(int_features['oldpeak'])],'slope':[float(int_features['slope'])],'ca':[int(int_features['ca'])],'thal':[int(int_features['thal'])]})
        r=get_data(final_features)
        print(r)
        r=json.dumps(r)
        result = make_response(r)
        result.headers['Content-Type'] = 'application/json'
        return result
     
 
def get_data(final_features):
   prediction=model.predict_proba(final_features)
   pred=(prediction[0][0]*0.842)
   if (pred<0.5):
        output='You have '+ str((1-pred)*100)+'%'+' possibility of having heart disease.Please take immediate action to cure your self.'
   elif (0.5<=pred<0.75):
        output='You have '+ str((1-pred)*100)+'%'+' possibility of having heart disease.You have to be careful and take some actions.'
   elif (pred>0.75):
        output='You have '+ str((1-pred)*100)+'%'+' possibility of having heart disease.You are safe'
 
    
   return {
       "fulfillmentText" : output,
       "intent": pred
        
   }
 
if __name__ == '__main__':
    port = int(os.getenv('PORT', 80))

    print ("Starting app on port %d" %(port))

    app.run(debug=True, port=port, host='0.0.0.0')
