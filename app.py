from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)


@app.route('/', methods=['GET','POST'])
def home():
    return render_template("index.html")  # return home.html

@app.route("/predict", methods=['POST'])

def predict():

    if request.method == 'POST':

      input_1 = request.form['gender']
      input_2 = request.form['SeniorCitizen']
      input_3 = request.form['Partner']
      input_4 = request.form['Dependents']
      input_5 = request.form['tenure']
      input_6 = request.form['PhoneService']
      input_7 = request.form['MultipleLines']
      input_8 = request.form['InternetService']
      input_9 = request.form['OnlineSecurity']
      input_10 = request.form['OnlineBackup']
      input_11 = request.form['DeviceProtection']
      input_12 = request.form['TechSupport']
      input_13 = request.form['StreamingTV']
      input_14 = request.form['StreamingMovies']
      input_15 = request.form['Contract']
      input_16 = request.form['PaperlessBilling']
      input_17 = request.form['PaymentMethod']
      input_18 = request.form['MonthlyCharges']
      input_19 = request.form['TotalCharges']

      model = pickle.load(open("telecom_model.sav", "rb"))

      sample_df = pd.read_csv("first_telc.csv")

      print(sample_df.isnull().sum())

      data = [[input_1,input_2,input_3,input_4,input_5,input_6,input_7,input_8,
               input_9,input_10,input_11,input_12,input_13,input_14,input_15,input_16,
               input_17,input_18,input_19]]

      print(data)

      new_df = pd.DataFrame(data, columns=[['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService',
                                            'MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
                                            'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract',
                                            'PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges']])
      print(new_df.head(1))

      final_df = pd.concat([sample_df, new_df], ignore_index=True)


      #new_df.columns = [x[0] for x in new_df.columns]

      labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
      final_df['tenure_group'] = pd.cut(final_df.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)

      final_df.drop(columns=['tenure'], axis=1, inplace=True)

      print(final_df)


      final_df_dummies = pd.get_dummies(final_df[['gender','SeniorCitizen','Partner','Dependents','PhoneService',
                                                       'MultipleLines','InternetService','OnlineSecurity','OnlineBackup'
                                                      ,'DeviceProtection','TechSupport','StreamingTV','StreamingMovies'
                                                      ,'Contract','PaperlessBilling','PaymentMethod','tenure_group']]
                                                      ,drop_first=True)

      print(final_df_dummies.shape)

      prediction = model.predict(final_df_dummies.tail(1))

      if prediction == 0:
            return render_template('index.html', prediction_texts="This customer is likely to CONTINUE services !")
      else:
            return render_template('index.html', prediction_text="The customer is likely to be CHURNED!!!")

    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)


