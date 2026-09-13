from pathlib import Path
import joblib

# folder used to store trained models
BASE_DIR = Path(__file__).resolve().parent
SAVED_MODEL_FOLDER = BASE_DIR / "saved_models"
SAVED_MODEL_FOLDER.mkdir(parents=True,exist_ok=True)

# find all models saved in the folder
def get_available_models():
    available_models = []
    # search for all .joblib model files and sort them by filename
    for model_path in sorted(SAVED_MODEL_FOLDER.glob("*.joblib")):
        try:
            saved_model = joblib.load(model_path)

            if isinstance(saved_model, dict):
                display_name = saved_model.get("model_name", model_path.stem)
            else:
                raise ValueError ("No model detected")

            available_models.append({"filename": model_path.name,"display_name": display_name,})

        except Exception as error:
            print(f"Could not read {model_path.name}: {error}")

    return available_models

# load the model file selected in the HTML dropdown
def load_selected_model(model_filename):
    # check that a model has been selected
    if not model_filename:
        raise ValueError("No model was selected.")

    model_folder_path = SAVED_MODEL_FOLDER
    # extract the filename
    model_name = Path(model_folder_path/model_filename).name
    # path to the selected model
    model_path = (SAVED_MODEL_FOLDER/model_name)
    # only select .joblib models
    if model_path.suffix.lower() != ".joblib":
        raise ValueError("The selected file is not a model.")

    # load the selected model
    model = joblib.load(model_path)

    return model