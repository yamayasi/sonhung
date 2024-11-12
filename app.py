import streamlit as st 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Thiết lập trang
st.title("Ứng dụng Hồi quy Hỗ trợ Vector (SVR)")

# Bước 1: Nhập dữ liệu
st.header("Bước 1: Nhập dữ liệu")
uploaded_file = st.file_uploader("Chọn tệp CSV hoặc Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Tải dữ liệu theo loại tệp
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        data = pd.read_excel(uploaded_file)
        
    st.write("Xem trước dữ liệu:", data.head())
    
    # Chọn cột mục tiêu để dự đoán
    target_column = st.selectbox("Chọn cột mục tiêu để dự đoán", data.columns)
    
    data["Số lượng mẫu"] = data.index + 1
    data = data[["Số lượng mẫu", target_column]]
    data = data.set_index("Số lượng mẫu")
    st.write("Xem trước dữ liệu:", data.head())

    # Đặt phạm vi ngày cho tập huấn luyện và kiểm tra
    st.header("Đặt phạm vi ngày cho tập huấn luyện và kiểm tra")
    train_start_dt = st.number_input("Ngày bắt đầu huấn luyện")
    test_start_dt = st.number_input("Ngày bắt đầu kiểm tra", value=9500)
    timesteps = st.number_input("Các bước thời gian", value=12)

    # Lọc dữ liệu theo phạm vi ngày
    try:
        train = data.copy()[(data.index >= train_start_dt) & (data.index < test_start_dt)][[target_column]]
        test = data.copy()[data.index >= test_start_dt][[target_column]]
        
        # Đảm bảo có đủ dữ liệu cho huấn luyện và kiểm tra
        if len(train) == 0 or len(test) == 0:
            st.warning("Hãy đảm bảo rằng phạm vi ngày đã chọn có dữ liệu cho cả huấn luyện và kiểm tra.")
        else:
            st.write("Xem trước dữ liệu huấn luyện:", train.head())
            st.write("Xem trước dữ liệu kiểm tra:", test.head())
            
            scaler = MinMaxScaler()
            train[target_column] = scaler.fit_transform(train)
            test[target_column] = scaler.transform(test)

            # Chuyển đổi thành mảng numpy
            train_data = train.values
            test_data = test.values

            train_data_timesteps = np.array([[j for j in train_data[i:i+timesteps]] for i in range(0, len(train_data)-timesteps+1)])[:,:,0]
            test_data_timesteps = np.array([[j for j in test_data[i:i+timesteps]] for i in range(0, len(test_data)-timesteps+1)])[:,:,0]

            x_train, y_train = train_data_timesteps[:, :timesteps-1], train_data_timesteps[:, [timesteps-1]]
            x_test, y_test = test_data_timesteps[:, :timesteps-1], test_data_timesteps[:, [timesteps-1]]
   
    except Exception as e:
        st.error(f"Lỗi khi chia tách dữ liệu: {e}")

# Bước 2: Cấu hình thông số của mô hình SVR
st.header("Bước 2: Cấu hình thông số của mô hình SVR")
C = st.number_input("C (Tham số điều chỉnh)", value=1000.0)
epsilon = st.number_input("Epsilon (Độ chính xác cho tiêu chuẩn dừng)", value=0.02)
gamma = st.selectbox("Gamma (Hệ số kernel)", [0.1, 0.01, 0.001])
kernel = st.selectbox("Loại kernel", ["linear", "poly", "rbf", "sigmoid"])

# Bước 3: Huấn luyện mô hình SVR
if uploaded_file is not None:
    # st.header("Bước 3: Huấn luyện mô hình SVR")

    # Khởi tạo và huấn luyện mô hình SVR
    model = SVR(
        C=C, 
        epsilon=epsilon,
        gamma=gamma,
        kernel=kernel
    )
    
    # Huấn luyện mô hình
    model.fit(x_train, y_train[:, 0])
    
    # Đánh giá mô hình
    y_train_pred = model.predict(x_train).reshape(-1, 1)
    y_test_pred = model.predict(x_test).reshape(-1, 1)

    y_train_pred = scaler.inverse_transform(y_train_pred)
    y_test_pred = scaler.inverse_transform(y_test_pred)

    y_train = scaler.inverse_transform(y_train)
    y_test = scaler.inverse_transform(y_test)

    mse = mean_squared_error(y_test, y_test_pred)
    # st.write(f"Lỗi Bình phương Trung bình trên Tập Kiểm tra: {mse}")


# Bước 4: Dự đoán
st.header("Bước 3: Dự đoán")
if uploaded_file is not None and len(train) > 0 and len(test) > 0:
    n_predictions = st.number_input("Số lượng dự đoán cần tạo", min_value=1, value=12)

    # Thiết lập ngày bắt đầu cho dự đoán vào lúc 9 giờ sáng
    start_date = st.date_input("Ngày bắt đầu cho dự đoán", value=datetime.datetime.now().date())
    start_dt = datetime.datetime.combine(start_date, datetime.time(9, 0))  # Thiết lập thời gian thành 9:00 sáng

    if st.button("Tạo dự đoán"):
        # Chuẩn bị dữ liệu để dự đoán
        data = data.copy().values
        data_predict = []

        for i in range(n_predictions):
            # Chọn các hàng 'timesteps' cuối cùng để dự đoán
            data_last_timesteps = data[-timesteps:]
            
            # Chuẩn hóa dữ liệu
            data_tf = scaler.transform(data_last_timesteps)
            
            # Chuyển đổi dữ liệu để phù hợp với yêu cầu đầu vào
            data_timesteps = np.array([[j for j in data_tf[i:i+timesteps]] for i in range(0, len(data_tf)-timesteps+1)])[:,:,0]
            
            X = data_timesteps[:, 1:]
            
            # Thực hiện dự đoán
            Y_pred = model.predict(X).reshape(-1, 1)
            
            # Chuẩn hóa ngược kết quả dự đoán
            Y_pred = scaler.inverse_transform(Y_pred)
            data_predict.append(Y_pred[0][0])  # Lưu dự đoán đầu tiên của mỗi vòng lặp

            # Cập nhật dữ liệu với giá trị dự đoán cho vòng lặp tiếp theo
            data = np.concatenate((data, Y_pred))
        
        # Tạo chỉ mục ngày giờ cho dự đoán với bước thời gian là 2 giờ
        prediction_dates = [start_dt + datetime.timedelta(hours=2 * i) for i in range(n_predictions)]
        
        # Tạo DataFrame cho dự đoán với chỉ mục thời gian
        predictions_df = pd.DataFrame(data_predict, index=prediction_dates, columns=[f"{target_column}_Dự đoán"])
        
        # Hiển thị DataFrame của dự đoán
        st.write(predictions_df)
        
        # Hiển thị biểu đồ dự đoán
        fig, ax = plt.subplots()
        ax.plot(predictions_df.index, predictions_df[f"{target_column}_Dự đoán"], marker='o', linestyle='-')
        # Format x-axis labels as dates
        # ax.tick_params(axis='x', labelrotation = 45)

        myFmt = mdates.DateFormatter('%d.%m.%Y %H:%M')
        ax.xaxis.set_major_formatter(myFmt)
        plt.gcf().autofmt_xdate()

        st.pyplot(fig)


# Bước 5: Dự đoán với trường hợp lũ lên nhanh
st.header("Bước 4: Dự đoán với trường hợp lũ lên nhanh")

if uploaded_file is not None and len(train) > 0 and len(test) > 0:
    # Nhập số lượng dự đoán và điểm bắt đầu/kết thúc
    n = st.number_input("Số lượng dự đoán cần tạo", value=12)
    n_start = st.number_input("Điểm bắt đầu dự đoán", min_value=0, value=1709)
    n_end = st.number_input("Điểm kết thúc dự đoán", min_value=n_start + 1, value=1745)
    timesteps = st.number_input("Bước thời gian", value=12)

    if st.button("Tạo dự đoán trong khoảng xác định"):
        # Trích xuất các giá trị tải dưới dạng mảng numpy từ n_start đến n_end
        data = data[n_start:n_end].copy().values  # Lấy dữ liệu trong khoảng n_start đến n_end
        data_predict = []
        
        # Đặt thời gian bắt đầu là 9 giờ sáng
        start_dt = datetime.datetime.combine(datetime.datetime.now().date(), datetime.time(9, 0))

        for i in range(int(n)):
            # Chọn các hàng 'timesteps' cuối cùng để dự đoán
            data_last_timesteps = data[-timesteps:]
            
            # Chuẩn hóa dữ liệu
            data_tf = scaler.transform(data_last_timesteps)
            
            # Chuyển đổi thành tensor 2D theo yêu cầu đầu vào của mô hình
            data_timesteps = np.array([[j for j in data_tf[i:i+timesteps]] for i in range(0, len(data_tf)-timesteps+1)])[:,:,0]
            
            X = data_timesteps[:, 1:]
            
            # Thực hiện dự đoán
            Y_pred = model.predict(X).reshape(-1, 1)
            
            # Chuẩn hóa ngược và lưu dự đoán
            Y_pred = scaler.inverse_transform(Y_pred)
            data_predict.append(Y_pred[0][0])  # Lưu dự đoán đầu tiên của mỗi vòng lặp
            
            # Cập nhật dữ liệu với giá trị dự đoán cho vòng lặp tiếp theo
            data = np.concatenate((data, Y_pred))
        
        # Tạo chỉ mục thời gian bắt đầu từ 9 giờ sáng với bước 2 giờ
        prediction_dates = [start_dt + datetime.timedelta(hours=2 * i) for i in range(int(n))]

        # Tạo DataFrame cho dự đoán với chỉ mục thời gian
        predictions_df = pd.DataFrame(data_predict, index=prediction_dates, columns=[f"{target_column}_Dự đoán"])

        # Hiển thị DataFrame của dự đoán
        st.write(predictions_df)
        
        # Vẽ biểu đồ đường cho dự đoán
        fig, ax = plt.subplots()
        ax.plot(predictions_df.index, predictions_df[f"{target_column}_Dự đoán"], marker='o', linestyle='-')
        myFmt2 = mdates.DateFormatter('%d.%m.%Y %H:%M')
        ax.xaxis.set_major_formatter(myFmt2)
        plt.gcf().autofmt_xdate()

        st.pyplot(fig)