from flask import Flask, render_template, request, redirect, url_for
from src.exception import CustomExceptions
from src.Pipelines.Prediction import PredictPipeline, CustomInputData
import sys
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/userdata', methods=['GET', 'POST'])
def user_data_prediction():
    try:
        if request.method == 'GET':
            return render_template('home.html')
        elif request.method == 'POST':
            # Remove leading zeros from the ID
            id_value = request.form.get('ID')
            if id_value is not None:
                id_value = id_value.lstrip('0')
                data = CustomInputData(
                    ID=int(id_value),
                    AGE=int(request.form.get('AGE')),
                    GENDER=int(request.form.get('GENDER')),
                    DRIVING_EXPERIENCE=int(request.form.get('DRIVING_EXPERIENCE')),
                    EDUCATION=int(request.form.get('EDUCATION')),
                    INCOME=int(request.form.get('INCOME')),
                    CREDIT_SCORE=float(request.form.get('CREDIT_SCORE')),
                    VEHICLE_OWNERSHIP=int(request.form.get('VEHICLE_OWNERSHIP')),
                    VEHICLE_YEAR=int(request.form.get('VEHICLE_YEAR')),
                    MARRIED=int(request.form.get('MARRIED')),
                    CHILDREN=int(request.form.get('CHILDREN')),
                    POSTAL_CODE=int(request.form.get('POSTAL_CODE')),
                    ANNUAL_MILEAGE=int(request.form.get('ANNUAL_MILEAGE')),
                    SPEEDING_VIOLATIONS=int(request.form.get('SPEEDING_VIOLATIONS')),
                    DUIS=int(request.form.get('DUIS')),
                    PAST_ACCIDENTS=int(request.form.get('PAST_ACCIDENTS')),
                    TYPE_OF_VEHICLE=int(request.form.get('TYPE_OF_VEHICLE'))
                )
                
                pred_dataframe1 = data.collect_data_user_to_dataframe()
                # return redirect(url_for('/check_approval', pred_dataframe=pred_dataframe1))
                # Pass pred_dataframe as a query parameter directly in the URL
                #return redirect(url_for('check_approval') + f'?pred_dataframe={pred_dataframe1}')
                return redirect(url_for('check_approval') + f'?pred_dataframe={pred_dataframe1.to_json()}')
    except Exception as e:
        raise CustomExceptions(e, sys)
    return render_template('home.html')

@app.route('/check_approval')
def check_approval():
    try:
        # df_data = user_data_prediction()  # Uncomment this line if needed
        pred_data_json = request.args.get('pred_dataframe')
        data = pd.read_json(pred_data_json)
        pred_data = pd.DataFrame(data, columns=["ID", "AGE", "GENDER", "DRIVING_EXPERIENCE", "EDUCATION", "INCOME",
                                                "CREDIT_SCORE", "VEHICLE_OWNERSHIP", "VEHICLE_YEAR", "MARRIED",
                                                "CHILDREN", "POSTAL_CODE", "ANNUAL_MILEAGE", "SPEEDING_VIOLATIONS",
                                                "DUIS", "PAST_ACCIDENTS", "TYPE_OF_VEHICLE"])

        pred_data.to_csv('json_user_data.csv')
        
        predict_pipe_line = PredictPipeline()
        results = predict_pipe_line.predict(pred_data)
        print(results)
        # Check if results is None or empty
        if results is None or not results:
            return render_template('prediction.html', results="N/A")

        # Check if the first element of results is '1'
        if results[0] == '1':
            return render_template('prediction.html', results=results[0], approval_status="approve")
        else:
            return render_template('prediction.html', results=results[0], approval_status="not approve")

    except Exception as e:
        raise CustomExceptions(e, sys)

if __name__ == "__main__":
    app.run(debug=True)
    
    #https://www.google.com/imgres?imgurl=https%3A%2F%2Finsurance-b2c-assets.s3.ap-south-1.amazonaws.com%2Fuploads%2Fnews%2Fimage%2Fmceu_4246681211664516426513_1664516426.jpg&tbnid=gYEHqCuK82RfxM&vet=12ahUKEwjArMLwuf2CAxVE3DgGHZ_iDsoQMygSegQIARBx..i&imgrefurl=https%3A%2F%2Fwww.insurancedekho.com%2Fcar-insurance%2Fnews%2Fhow-to-check-car-insurance-expiry-date.htm&docid=---51iquxCutvM&w=930&h=620&q=vehicle%20insurance%20check%20images&ved=2ahUKEwjArMLwuf2CAxVE3DgGHZ_iDsoQMygSegQIARBx
