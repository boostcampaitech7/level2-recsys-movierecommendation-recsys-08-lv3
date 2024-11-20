import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


def catboost_train(args):

    X_train, y_train = args.train_data.drop(columns=["rating"]), args.train_data[["rating"]]

    model = CatBoostClassifier(
        iterations=args.iterations,
        learning_rate=args.lr,
        depth=args.depth,
        loss_function="Logloss",
        task_type="GPU",
        devices=args.gpu_devices,
        verbose=args.verbose,
    )

    model.fit(X_train, y_train)

    return model


def catboost_valid_evaluate(args, model):
    train_data = args.train_data
    val = args.valid_data
    recall_sum = 0

    for user, group in train_data.groupby("user"):
        candidate_each_user = group[group["rating"] == 0]

        prediction_probability = model.predict_proba(candidate_each_user.drop(columns=["rating"]))[:, 1]

        top_10_indices = np.argsort(prediction_probability)[-10:]
        top_10_items = candidate_each_user.iloc[top_10_indices]["item"].tolist()
        valid_items = val.loc[val["user"] == user, "item"].tolist()

        correct_num = len(set(top_10_items) & set(valid_items))
        denominator = min(10, len(valid_items))
        recall = correct_num / denominator

        recall_sum += recall

    user_number = train_data["user"].nunique()
    recall_K = recall_sum / user_number

    return recall_K


def catboost_predict(args):
    combined_data = pd.concat([args.train_data, args.valid_data])
    combined_data = combined_data.sort_values(by="user").reset_index(drop=True)

    X_final, y_final = combined_data.drop(columns=["rating"]), combined_data[["rating"]]

    model = CatBoostClassifier(
        iterations=args.iterations,
        learning_rate=args.lr,
        depth=args.depth,
        loss_function="Logloss",
        task_type="GPU",
        devices=args.gpu_devices,
        verbose=args.verbose,
    )

    model.fit(X_final, y_final)

    recommended_items = []

    for user, group in combined_data.groupby("user"):
        candidate_each_user = group[group["rating"] == 0]

        prediction_probability = model.predict_proba(candidate_each_user.drop(columns=["rating"]))[:, 1]

        top_10_indices = np.argsort(prediction_probability)[-10:]
        top_10_items = candidate_each_user.iloc[top_10_indices]["item"].tolist()

        for item in top_10_items:
            recommended_items.append([user, item])

    recommended_df = pd.DataFrame(recommended_items, columns=["user", "item"])

    recommended_df.to_csv("recommended_items.csv", index=False)
