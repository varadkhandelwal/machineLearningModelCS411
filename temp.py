from flask import Flask, render_template, request
import pickle
import numpy as np
import math
from flask_cors import CORS
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt, mpld3

#### Getting all the arguments
cat = "Art"#request.args.get('category').strip()
month = 3#int(request.args.get('month').strip())
backers = 0#int(request.args.get('backers').strip())
country = 0#int(request.args.get('country').strip())
money_raised = 10000#int(request.args.get('money_raised').strip())

#### Converting cat into oneHot Model to feed it into model
cat  = "_" + cat
cat_list = ['_Art', '_Comics', '_Crafts', '_Dance', '_Design', '_Fashion', '_Film & Video', '_Food', '_Games', 
    '_Journalism', '_Music', '_Photography', '_Publishing', '_Technology',
    '_Theater']
cat_index = cat_list.index(cat)
cat_feed = [0]*len(cat_list)
cat_feed[cat_index] = 1

#### Creating the input for model
input_ = [month, backers, country]
input_.extend(cat_feed)
input_ = np.array(input_).reshape(1, -1)

#### Loading the model 
lr = pickle.load(open( "model.pkl", "rb" ))
money_predicted = lr.predict(input_)

#### Adjusting Money predicted to reflect current money raised
while(money_predicted < money_raised):
    money_predicted *= math.log(money_raised)

if(money_predicted > money_raised): 
    money_predicted += money_raised

#### Projecting money over a six year plane, using the economic growth rate to 
#### discount it
money_predictions = []
for i in range(5, -1, -1):  
    money_predictions.append(-np.pv(rate=0.60, 
                nper=i, pmt=-(money_predicted-money_raised)*0.07, 
                fv=money_predicted-money_raised)[0] + money_raised)

print(money_predictions)
#### Generating plot
d = {'Year':[2019, 2020, 2021, 2022, 2023, 2024],'Money_raised':money_predictions}
df = pd.DataFrame(d)

sns.set(style="darkgrid")
sns.lineplot(x="Year", y="Money_raised", 
            data=df, markers=True)

fig, ax = plt.subplots()
result = mpld3.fig_to_html(fig)

print(result)