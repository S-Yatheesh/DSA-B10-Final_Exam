import os
import pickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSOR_PATH = os.path.join(BASE_DIR, 'preprocessor.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'shopping_model.pkl')


try:
	with open(PREPROCESSOR_PATH, 'rb') as f:
		preprocessor = pickle.load(f)
except Exception as e:
	raise RuntimeError(f"Failed to load preprocessor from {PREPROCESSOR_PATH}: {e}")

try:
	with open(MODEL_PATH, 'rb') as f:
		model = pickle.load(f)
except Exception as e:
	raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")


FORM_FIELDS = [
	'Age',
	'Gender',
	'Occupation_Status',
	'Country_Region',
	'Monthly_Income_Range',
	'Online_Shopping_Frequency',
	'Preferred_Platform',
	'Primary_Device',
	'Top_Spending_Category',
	'Review_Influence_Score',
	'Social_Ads_Influence_Score',
	'Price_Comparison_Frequency',
	'Discount_Importance_Score',
	'Follows_Brands_On_Social',
	'Makes_Shopping_List',
	'Impulse_Purchase_Frequency',
	'Return_Frequency',
	'Avg_Monthly_Spend_NonEssentials',
	'Preferred_Payment_Method',
	'Uses_BNPL_Installments',
	'Online_Shopping_Satisfaction_Score',
	'Regret_After_Purchase_Frequency'
]

# Fields that should be coerced to numeric types
INT_FIELDS = {'Age'}
FLOAT_FIELDS = {
	'Review_Influence_Score', 'Social_Ads_Influence_Score', 'Price_Comparison_Frequency',
	'Discount_Importance_Score', 'Impulse_Purchase_Frequency', 'Return_Frequency',
	'Avg_Monthly_Spend_NonEssentials', 'Online_Shopping_Satisfaction_Score',
	'Regret_After_Purchase_Frequency'
}


@app.route('/', methods=['GET', 'POST'])
def index():
	prediction = None
	form_data = {}

	if request.method == 'POST':
		# Collect and coerce form values into a dict matching the original schema
		for key in FORM_FIELDS:
			raw = request.form.get(key)
			if raw is None or raw == '':
				form_data[key] = ''
				continue

			if key in INT_FIELDS:
				try:
					form_data[key] = int(raw)
				except Exception:
					form_data[key] = None
			elif key in FLOAT_FIELDS:
				try:
					form_data[key] = float(raw)
				except Exception:
					form_data[key] = None
			else:
				form_data[key] = raw

		# Build a single-row DataFrame in the same column order as training schema
		df = pd.DataFrame([form_data], columns=FORM_FIELDS)

		# Transform and predict
		try:
			X_transformed = preprocessor.transform(df)
			y_pred = model.predict(X_transformed)
			# Normalize prediction to a string label
			if hasattr(y_pred, '__len__'):
				pred_label = y_pred[0]
			else:
				pred_label = y_pred
			prediction = str(pred_label)
		except Exception as e:
			prediction = f"Prediction error: {e}"

	return render_template('index.html', prediction=prediction, form_data=form_data)


if __name__ == '__main__':
	app.run(debug=True)

