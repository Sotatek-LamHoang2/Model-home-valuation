import datetime
import pandas as pd

def cleanDataTK():
    data = pd.read_excel("DataTKbatch1.xlsx")
    data = data.drop(columns=["Địa chỉ nhà", "Ngõ"])

    for column in ["Quận", "Phường/xã", "Đường phố"]:
        data[column] = data[column].str.lower().str.strip()

    data["Diện tích trên sổ (m²)"] = pd.to_numeric(data["Diện tích trên sổ (m²)"], errors="coerce")
    data = data.dropna(subset=["Diện tích trên sổ (m²)", "Số tầng", "Độ rộng mặt tiền trên sổ (m)", ])

    data["Vỉa hè"] = data["Vỉa hè"].replace("Không", 0)
    data["Vỉa hè"] = data["Vỉa hè"].replace("Có - 3m", 3)
    data["Vỉa hè"] = pd.to_numeric(data["Vỉa hè"], errors="coerce")

    data["Độ rộng đường nhỏ nhất khi đi vào nhà"] = pd.to_numeric(data["Độ rộng đường nhỏ nhất khi đi vào nhà"], errors="coerce")
    data = data.dropna(subset=["Độ rộng đường nhỏ nhất khi đi vào nhà"])

    data["Độ rộng mặt thoáng bên cạnh"] = data["Độ rộng mặt thoáng bên cạnh"].replace('Không có', 0)
    data["Độ rộng mặt thoáng bên cạnh"] = pd.to_numeric(data["Độ rộng mặt thoáng bên cạnh"], errors="coerce")
    data = data.dropna(subset=["Độ rộng mặt thoáng bên cạnh"])

    data["Độ rộng mặt thoáng sau nhà"] = data["Độ rộng mặt thoáng sau nhà"].replace('Không có', 0)
    data["Độ rộng mặt thoáng sau nhà"] = pd.to_numeric(data["Độ rộng mặt thoáng sau nhà"], errors="coerce")
    data = data.dropna(subset=["Độ rộng mặt thoáng sau nhà"])

    data["Dòng tiền thuê trong 1 năm"] = pd.to_numeric(data["Dòng tiền thuê trong 1 năm"], errors="coerce")

    # for column in data.columns:
    #     print(column)
    #     print(data[column].dtype)
    #     print(data[column].unique())
    #     print("-----------------------------------------------------------")

    print(data.head())
    print(data.info())

    return data

if __name__ == "__main__":
    cleanDataTK()