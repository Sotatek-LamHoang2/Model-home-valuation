import json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from cleanData import cleanDataTK
from cleanDataVN import cleanDataVN

def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = (~np.isnan(y_true)) & (y_true != 0)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def main() -> None:
    # 1️ Đọc dữ liệu
    # data_path = "DataCleaning.xlsx"
    # df = pd.read_excel(data_path)
    # df = df.replace([np.inf, -np.inf], np.nan)
    df = cleanDataTK()

    # 2️ Chuẩn bị dữ liệu
    df = df.drop_duplicates().reset_index(drop=True)
    X = df.drop(columns=["GIÁ CHỐT"])
    y = pd.to_numeric(df["GIÁ CHỐT"], errors="coerce")

    valid_mask = ~y.isna()
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)

    # 3️ CatBoost tự xử lý NaN — chỉ cần giữ categorical dạng string
    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    X[cat_features] = X[cat_features].astype(str)

    # 4️ Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    # 5️ Khai báo mô hình (cấu hình cơ bản, dễ hiểu)
    from catboost import CatBoostRegressor
    # load params đã huấn luyện từ trước
    with open("best_catboost_params.json", "r") as f:
        best_params = json.load(f)
    model = CatBoostRegressor(**best_params)
    model = CatBoostRegressor(
        iterations=1649,
        learning_rate=0.061544271574852594,
        depth=6,
        l2_leaf_reg=0.017233365408298063,
        random_strength=0.082487309197701,
        bagging_temperature=0.46493700396157595,
        border_count=128,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=42,
        verbose=200,
    )


    # 6️ Huấn luyện
    model.fit(
        X_train,
        y_train,
        cat_features=cat_features if cat_features else None,
        eval_set=(X_test, y_test),
    )

    # 7️ Đánh giá
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = safe_mape(y_test.to_numpy(), y_pred)

    print("\n--- Kết quả đánh giá ---")
    print(f"MAE: {mae:,.2f}")
    print(f"R²: {r2:.3f}")
    print(f"MAPE: {mape:.2f}%")

    # 8️ Lưu model + danh sách cột
    model.save_model("catboost_model.cbm")
    print("✅ Mô hình đã được lưu vào catboost_model.cbm")


if __name__ == "__main__":
    main()
