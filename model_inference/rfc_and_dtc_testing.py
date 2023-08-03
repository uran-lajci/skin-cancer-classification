import pandas as pd
import joblib
import json


def load_test_data():
    test_dataset_path = f"dataset/preprocessed_test_dataset.csv"
    test_df = pd.read_csv(test_dataset_path)
    x_test = test_df.drop(['image_id', 'lesion_type'], axis=1)
    image_ids = test_df['image_id']
    return x_test, image_ids


def load_model(algorithm_choice):
    while algorithm_choice not in ["RFC", "DTC"]:
        print(f"Invalid algorithm choice: \"{algorithm_choice}\". Please choose either 'RFC' or 'DTC'.")
        algorithm_choice = input("Choice: ")

    model_path = f'model_training/models/{algorithm_choice}_roc_model.pt'
    rfc = joblib.load(model_path)
    return rfc


def predict_using_model(model, x_test, image_ids):
    y_pred = model.predict(x_test)

    # Create a mapping dictionary for lesion type labels
    lesion_type_mapping = {
        0: "BKL",
        1: "NV",
        2: "DF",
        3: "MEL",
        4: "VASC",
        5: "BCC",
        6: "AKIEC"
    }
    output = []
    for image_id, lesion_type in zip(image_ids, y_pred):
        output.append({
            "image_id": str(image_id),  
            "lesion_type": lesion_type_mapping[int(lesion_type)]  
        })
    return output


def save_output_to_json(output, algorithm_choice):
    output_path = f"model_inference/{algorithm_choice}_output.json"
    with open(output_path, "w") as f:
        json.dump(output, f)

    
if __name__ == "__main__":
    x_test, image_ids = load_test_data()
    algorithm_choice = input("""Write RFC for Random Forest Classifier,
    write DTC for Decision Tree Classifier,
    Choice: """)

    model = load_model(algorithm_choice)
    output = predict_using_model(model, x_test, image_ids)
    save_output_to_json(output, algorithm_choice)
    print("Completed the model inference.")