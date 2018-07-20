import pickle
import numpy as np

model = pickle.load(open("models/sklearn_rf.pkl","rb"))

def predict(args):
  print(args["feature"])
  account=np.array(args["feature"].split(",")).reshape(1,-1)
  return {"result" : model.predict(account)[0]}
  
predict({
  "feature":"0,1,2,3,4,5,6,7,8,9,10,11"
})