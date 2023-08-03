import pandas as pd
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def get_dataset_choice():
    choices = ["smote", "no_balance", "roc"]
    while True:
        dataset_choice = input("""Write no_balance for the unbalanced dataset,
        write smote for the SMOTE dataset,
        write roc for the Random Oversampling dataset,
        Choice: """)
        if dataset_choice in choices:
            return dataset_choice
        else:
            print(f"You wrote a wrong input \"{dataset_choice}\" for the dataset choice.")


def get_algorithm_choice():
    choices = ["RFC", "DTC"]
    while True:
        algorithm_choice = input("""Write RFC for Random Forest Classifier,
        write DTC for Decision Tree Classifier,
        Choice: """)
        if algorithm_choice in choices:
            return algorithm_choice
        else:
            print(f"You wrote a wrong input \"{algorithm_choice}\" for the algorithm choice.")


def load_data(dataset_choice):
    train_df = pd.read_csv(f"dataset/{dataset_choice}_preprocessed_train_dataset.csv")
    valid_df = pd.read_csv(f"dataset/preprocessed_validation_dataset.csv")
    return train_df, valid_df


def separate_data(train_df, valid_df):
    x_train = train_df.drop(['lesion_type'], axis = 1)
    y_train = train_df['lesion_type']
    x_valid = valid_df.drop(['lesion_type', 'image_id'], axis = 1)
    y_valid = valid_df['lesion_type']

    # this attribute is missing in the valid dataset
    # so we add it with 0.0 values
    if 'localization_acral' not in x_valid.columns:
        x_valid['localization_acral'] = 0.0

    x_valid = x_valid.reindex(columns = x_train.columns)

    return x_train, y_train, x_valid, y_valid


def build_and_fit_model(algorithm_choice, x_train, y_train):
    if algorithm_choice == "RFC":
         model = RandomForestClassifier(random_state=42)
         model.fit(x_train, y_train)

    elif algorithm_choice == "DTC":
        model = DecisionTreeClassifier(random_state=42)
        model.fit(x_train, y_train)
    
    return model


def evaluate_model(model, x_valid, y_valid):
    y_pred = model.predict(x_valid)

    accuracy = accuracy_score(y_valid, y_pred)
    precision = precision_score(y_valid, y_pred, average='weighted')
    recall = recall_score(y_valid, y_pred, average='weighted')
    f1 = f1_score(y_valid, y_pred, average='weighted')

    print('Evaluation of the Model')
    print('Accuracy', accuracy)
    print('Precision', precision)
    print('Recall', recall)
    print('F1-Score', f1)

    
if __name__ == "__main__":
    dataset_choice = get_dataset_choice()
    train_df, valid_df = load_data(dataset_choice)

    x_train, y_train, x_valid, y_valid = separate_data(train_df, valid_df)

    algorithm_choice = get_algorithm_choice()

    model = build_and_fit_model(algorithm_choice, x_train, y_train)

    evaluate_model(model, x_valid, y_valid)

    joblib.dump(model, f"model_training/models/{algorithm_choice}_{dataset_choice}_model.pt")
    print(f"Saved the {algorithm_choice} model.")