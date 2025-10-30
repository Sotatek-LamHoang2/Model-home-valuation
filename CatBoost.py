import json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import optuna

from cleanData import cleanDataTK

def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = (~np.isnan(y_true)) & (y_true != 0)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

# 2. Định nghĩa hàm "objective" cho Optuna
def objective(trial: optuna.Trial, X_train, y_train, X_test, y_test, cat_features) -> float:
    
    # Định nghĩa không gian tìm kiếm cho các tham số
    params = {
        "iterations": trial.suggest_int("iterations", 500, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_categorical("border_count", [32, 64, 128, 255]),
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "random_seed": 42,
        "verbose": False,
    }

    # Khởi tạo mô hình với các tham số được 'trial' đề xuất
    model = CatBoostRegressor(**params)

    # Huấn luyện mô hình
    model.fit(
        X_train,
        y_train,
        cat_features=cat_features if cat_features else None,
        eval_set=(X_test, y_test),
        early_stopping_rounds=100, # Dừng sớm sau 100 vòng nếu MAE không giảm
        verbose=False,
    )

    # Đánh giá trên tập test
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    return mae # Optuna sẽ cố gắng 'minimize' giá trị này


def main() -> None:
    # 1️ Đọc dữ liệu
    df = cleanDataTK()

    # 2️ Chuẩn bị dữ liệu
    df = df.drop_duplicates().reset_index(drop=True)
    X = df.drop(columns=["GIÁ CHỐT"])
    y = pd.to_numeric(df["GIÁ CHỐT"], errors="coerce")

    valid_mask = ~y.isna()
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    X[cat_features] = X[cat_features].astype(str)

    # 4️ Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    
    # 5 setup tham số huấn luyện
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_test, y_test, cat_features),
        n_trials=100
    )

    print("✅ Tìm kiếm hoàn tất!")
    print(f"Giá trị MAE tốt nhất: {study.best_value:,.2f}")
    print("Bộ tham số tối ưu (Best params):")
    print(study.best_params)
    
    # 6️⃣ Huấn luyện mô hình CUỐI CÙNG với bộ tham số tốt nhất
    print("\n--- Huấn luyện mô hình cuối cùng với tham số tốt nhất ---")
    best_params = study.best_params
    
    # Thêm lại các tham số cố định
    best_params.update({
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "random_seed": 42,
        "verbose": 200,
    })
    
    final_model = CatBoostRegressor(**best_params)
    
    # Fit lần cuối trên toàn bộ tập train, không cần early stopping
    final_model.fit(
        X_train,
        y_train,
        cat_features=cat_features if cat_features else None,
        eval_set=(X_test, y_test),
    )

    # 7️ Đánh giá mô hình cuối cùng
    y_pred = final_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = safe_mape(y_test.to_numpy(), y_pred)

    print("\n--- Kết quả đánh giá (Mô hình cuối cùng) ---")
    print(f"MAE: {mae:,.2f}")
    print(f"R²: {r2:.3f}")
    print(f"MAPE: {mape:.2f}%")

    # 8️ Lưu model
    final_model.save_model("catboost_model_tuned.cbm")
    print("✅ Mô hình TỐI ƯU đã được lưu vào catboost_model_tuned.cbm")

    # 9 Lưu lại thông số tốt nhât
    with open("best_catboost_params.json", "w") as f:
        json.dump(best_params, f, indent=4)
    print("✅ Đã lưu best_params vào 'best_catboost_params.json'")


if __name__ == "__main__":
    main()