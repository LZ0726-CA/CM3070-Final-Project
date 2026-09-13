# Flask reference: https://flask.palletsprojects.com/en/stable/quickstart/
from flask import Flask, render_template, request
from models import get_available_models
from prediction import run_prediction, scan_index
from features import FEATURE_GROUPS
from training import run_training

# create flask application
app = Flask(__name__)

# create the default values in html page
def create_page_data():
    return {
        # prediction section values
        "ticker": "",
        "selected_model_id": "",
        "selected_model": None,
        "prediction": None,
        "prediction_date": None,
        "predict_direction": None,
        "ai_explanation": None,
        "prediction_error": None,

        # scanning section values
        "selected_scan_index": "",
        "selected_scan_model": "",
        "scan_results": None,
        "scan_error": None,
        "scan_completed": False,

        # training section values
        "new_model_name": "",
        "selected_index": "SP500",
        "training_start_date": "",
        "training_end_date": "",
        "selected_features": [],

        # training results
        "training_message": None,
        "training_error": None,
        "training_metrics": None,
        "saved_model_filename": None,
 
        # selection values
        "available_models": get_available_models(),
        "feature_groups": FEATURE_GROUPS,
    }

# route
@app.route("/", methods=["GET", "POST"])
def index():
    page_data = create_page_data()

    if request.method == "POST":
        action = request.form.get("action","").strip()
        # run prediction when predict button is pressed
        if action == "predict":
            run_prediction(page_data)
        # scan index when scan button is pressed
        elif action == "scan":
            scan_index(page_data)
        # run training when train button is pressed
        elif action == "train":
            run_training(page_data)
        else:
            raise RuntimeError("Unknown action")

        # refresh the available model list
        page_data["available_models"] = get_available_models()

    # send data to jinja template
    return render_template("app.html",**page_data)

# run flask application
if __name__ == "__main__":
    app.run(
        debug=True,
        use_reloader=False
    )