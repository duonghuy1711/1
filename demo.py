import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# tạo class implement SVM algorithm
class SVM():
    # khởi tạo các siêu tham số
    def __init__(self, learning_rate=1e-3, lambda_param=1e-2, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
    # Khởi tạo weights và bias
    def _init_weights_bias(self, X):
        n_features = X.shape[1]
        self.w = np.zeros(n_features)
        self.b = 0
    # lấy giá trị phân loại tương ứng  với giá trị của hàm mục tiêu
    def _get_cls_map(self, y):
        return np.where(y <= 0, -1, 1)
    # kiểm tra dấu của điểm 
    def _satisfy_constraint(self, x, idx):
        linear_model = np.dot(x, self.w) + self.b 
        return self.cls_map[idx] * linear_model >= 1
    # tính gradient
    def _get_gradients(self, constrain, x, idx):
        if constrain:
            dw = self.lambda_param * self.w
            db = 0
            return dw, db
        
        dw = self.lambda_param * self.w - np.dot(self.cls_map[idx], x)
        db = - self.cls_map[idx]
        return dw, db
    #update trọng số
    def _update_weights_bias(self, dw, db):
        self.w -= self.lr * dw
        self.b -= self.lr * db
    # huấn luyện mô hình
    def fit(self, X, y):
        self._init_weights_bias(X)
        self.cls_map = self._get_cls_map(y)

        for _ in range(self.n_iters):
            for idx, x in enumerate(X):
                constrain = self._satisfy_constraint(x, idx)
                dw, db = self._get_gradients(constrain, x, idx)
                self._update_weights_bias(dw, db)
    # dự đoán đầu ra
    def predict(self, X):
        estimate = np.dot(X, self.w) + self.b
        prediction = np.sign(estimate)
        return np.where(prediction == -1, 0, 1)
# huấn luyện trên dataframe train và test
def run(df_train, df_test):
    #tách dữ liệu và nhãn
    X_train = df_train.iloc[:,:-1].values
    Y_train = df_train.iloc[:,-1].values
    X_test = df_test.iloc[:,:-1].values
    Y_test = df_test.iloc[:,-1].values
    label = np.unique(Y_train)
    #print(label)
    # mã hoá nhãn
    Y_train = np.where(Y_train == 'Besni', 0, 1)
    Y_test = np.where(Y_test == 'Besni', 0, 1)
    # Model training
    # biến lưu error toàn cục nhỏ nhất
    global error_min 
    model = SVM(learning_rate=1e-3, lambda_param=1e-2, n_iters=10)
    model.fit( X_train, Y_train )
      # Prediction on test set
    
    Y_pred = model.predict( X_test )
    res = [int(x) for x in Y_pred]
    Y_pred = np.around(res, decimals=1)
    error_min=1e5
    import pickle
    # calculate MAE
    from sklearn import metrics
    error_train = metrics.mean_absolute_error(Y_train, model.predict( X_train ))
    error_test = metrics.mean_absolute_error(Y_test, Y_pred)
    # cập nhật  error_min và lưu lại model tốt nhất
    if (error_min > error_train + error_test):
        error_min = error_test + error_train
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)
        #print('min error is',error_min)
    # tính toán các độ đo trên bộ kiểm tra dựa trên y_pred và y_test
    accuracy = metrics.accuracy_score(Y_test, Y_pred)
    precision = metrics.precision_score(Y_test, Y_pred,zero_division=0)
    recall = metrics.recall_score(Y_test, Y_pred)
    f1 = metrics.f1_score(Y_test, Y_pred)
    #print(metrics.confusion_matrix(Y_test, Y_pred))
    return accuracy,precision,recall,f1
# triển khai kỹ thuật k-fold
def k_fold(df, k = 3):
    folds = list()
    fold_size = int(df.shape[0] / k)
    start_index = 0
    # chia df ban đầu thành các subset có kích thước fold_size, subset cuối cùng có thể có số phần từ <fold_size
    for i in range(k):
      if (start_index + fold_size <= df.shape[0]):
          folds.append(df.loc[start_index: start_index + fold_size])
          start_index = start_index + fold_size
      else:
          folds.append(df.loc[start_index: df.shape[0]])
    return folds
# chia tập train test
def train_test_split(df, frac=0.3):
    
    # get random sample 
    test = df.sample(frac=frac, axis=0)

    # get everything but the test sample
    train = df.drop(index=test.index)

    return train, test

#load dữ liệu từ file .xlsx
df = pd.read_excel("Raisin_Dataset.xlsx")
#  xử lý các ô dữ liệu lỗi (NaN)
df  = df.fillna(0)
df = df.sample(frac=1).reset_index(drop=True)
k = 3
error_min = 1e5
#folds chứa danh sách các subset của df ban đầu
folds = k_fold(df,k)
frac = 0.3
df_train, df_test = train_test_split(df,frac)
# thử nghiệm SVM không áp dựng k-fold cross validation
acurracy,precision,recall,f1 = run(df_train, df_test)
acurracy=round(acurracy,2)
precision=round(precision,2)
recall=round(recall,2)
f1=round(f1,2)
#print("Accuracy: ",acurracy,"\nPrecision: ",precision,"\nRecall: ",recall,"\nF1: ",f1,"\n")
# thử nghiệm SVM kết hợp k-fold cross validation
for i in range(k):
    fold_test = folds[i]
    fold_train = []
    for j in range(k):
        if j != i:
            fold_train.append(folds[j])
    fold_train = pd.concat(fold_train)
    acurracy,precision,recall,f1=run(fold_train, fold_test)
    acurracy=round(acurracy,2)
    precision=round(precision,2)
    recall=round(recall,2)
    f1=round(f1,2)
    #print("Accuracy: ",acurracy,"\nPrecision: ",precision,"\nRecall: ",recall,"\nF1: ",f1,"\n")
