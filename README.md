# DSAI-HW2-2022
The homework-2 of the [NCKU](https://www.ncku.edu.tw/index.php?Lang=en) course which named Competitions in [**D**ata **S**ciences and **A**rtificial **I**ntelligence](http://class-qry.acad.ncku.edu.tw/syllabus/online_display.php?syear=0110&sem=2&co_no=P75J000&class_code=).<br>
* [The example code](https://github.com/NCKU-CCS/DSAI-HW2-2021).
* [The profit calculator (評分程式，原版)](https://github.com/NCKU-CCS/StockProfitCalculator).
* [The profit calculator kai (評分程式，本人修改為方便建置的 conda 版)](https://github.com/kuihao/StockProfitCalculator_CondaVer)
## PROBLEM DESPRICTION
### TASK
Given a series of stock prices, including **daily open (opening price), high (highest share price), low (lowest share price, and close (close price)**, decide your daily action and make your best profit for the future trading. Can you beat the simple **“buy-and-hold”** strategy?
### RULE
Given 20 days of stock market operations, only one operation can be performed each day:<br>
0 means no action, <br>
1 means buy one stock, <br>
-1 means sell one stock.<br>
Short selling is permitted. <br>
**The maximum number of stocks in the stock account for this operation is one stock or one short sale.** If you have bought one stock, you need to wait for another day to sell it, and then wait for another day to buy it back, so **you cannot buy and sell on the same day**.
### GOAL
Maximize revenue in 20 days.
### NOTICE
* **每讀完一天的資料，就必須輸出隔天是否要購買股票，輸出後才能再讀下一天的資料，以此類推。**
    * E.g.:
        * D = 新的一天股票資料產生
        * A = 模型執行的動作 (-1, 0, 1)
        * **合法的順序： DADADADADA**
* **違反下列規則 0 分計算:**
    * 使用非法的測試資料讀取方式，E.g.: DDDDDADDA
    * 一次讀完全部股票資料才輸出是否要購買
    * 修改輸出結果
    * 輸出結果數量不正確
    * Your code is the same as other classmate’s.
    * Your Github repo does not contain your code.
    * You do not submit the homework before the deadline.
    * **The training dataset you used is not provided by TA. (言下之意，不能更改/新增 training data)** 
    * Your code can get data through the Internet during training or testing. (不能使用爬蟲) 
    * Your code terminates during testing.
### Q&A
**Q:** Can I buy one stock at first day (status = 1), the sell and short selling of it at the same sencond day (status = -1)?<br>
**A:** Outputs 描述的是 "操作" 而非狀態，1 為買入一張股票，0 為不操作，-1 為賣出一張股票<br>
也就是說，如果output的前兩天都是1即代表，你同時持有兩張股票是種違規操作，而你的例子<br>
Day 1 outputs "1"<br>
Day 2 outputs "-1"<br>
代表你最後不持有任何股票
## NSTRUCTIONS FOR USE
(環境建置及程式使用說明)
### Prerequisite
- [conda](https://docs.conda.io/en/latest/index.html)
- To save the requirements.txt
```shell 
conda list --export > requirements.txt
```
or
```shell 
pip freeze > requirements.txt  
```

### Build Eev.
create an python 3.6 env.
```sh 
conda create -n StockProfitCalculator-py36 python=3.6.4
```
To activate Env. in **linux**:
```sh 
conda activate StockProfitCalculator-py36
```
To activate Env. in **windows**:
```sh 
activate StockProfitCalculator-py36
```
### Install Dependency
```
pip install -r requirements.txt
```
## HIGHLIGHT (PROPOSED METHOD)
### Observeation
Draw candlestick to observe the trend of input data<br>
Training data 20 days<br>
![candlestick_20_day](./img/20Day_candlestick.png)
Training data all time<br>
![candlestick_all_time](./img/allday_candlestick.png)
Observe the frequency domain<br>
![FFT](./img/fft.png)
### Data Preprocess
Calculate moving average: Opening price<br>
![MA_OpeningPrice](./img/MA_OpenPrice.jpg)
Calculate moving average: Closing price<br>
![MA_ClosingPrice](./img/MA_ClosePrice.jpg)
### Model (GRU)
**Model 1:** Input data for the previous **120 days (14 features)** to predict **tomorrow's opening and closing prices**
* ![Model-1 Struct.](./img/model1_struct.png)

**Model 2:** Input data for the previous **120 days (14 features)** to predict the **opening** and **closing prices** for **tomorrow and the day after**
* ![Model-2 Struct.](./img/model2_struct.png)

## LISTING 
(explain the function of each file)
* **app.py 主程式**，使用方式如下: (與範例程式相同) 
```shell 
python app.py --training training_data.csv --testing testing_data.csv --output output.csv
```
* app_v2_Trader.ipynb，Data experiment, main model training and tuning. 
* **app_v3_Only_Trader.ipynb**，Traing Part 的完成版，去除多餘函式並改寫成方便套用於 app.py 的格式
* ./model，儲存多個實驗模型，穩定版本為 model(1/2)_2022_04_21__21_08_52.h5
* ./log，對應實驗模型的實驗記錄
* training_data.csv、testing_data.csv 作業提供的 raw data
* output.csv 最終結果 (每次執行程式時，會自動把舊的 output.csv 刪除)
## ENVIRONMENT AND EXPERIMENT DESIGN
* Python: 3.6 (3.6.4)
* OS: Ubuntu 20.04 amd64
* GPU: Nvidia Geforce GTX 1070
* [Dataset](https://www.nasdaq.com/market-activity/stocks/ibm), [Training Data](https://www.dropbox.com/s/uwift61i6ca9g3w/training.csv?dl=0), [Testing Data](https://www.dropbox.com/s/duqiffdpcadu6s7/testing.csv?dl=0)
## TRAINING RESULT
* Model 1: Loss and Prediction
    * Loss value training trend
        * ![Model-1 loss](./log/2022_04_21__22_23_05/model1_loss.png)
    * Prediction
        * ![Model-1 Prediction](./log/2022_04_21__22_23_05/model1_predict_vs_truth.png)
        * The black line is the Train set y value, which is the actual opening price on the next day, for a total of 1248 days
        * The orange line is the model prediction, which is the actual opening price on the next day, for a total of 1248 days
* Model 2: Loss and Prediction
    * Loss value training trend
        * ![Model-2 loss](./log/2022_04_21__22_23_05/model2_loss.png)
    * Prediction
        * ![Model-2 Prediction](./log/2022_04_21__22_23_05/model2_predict_vs_truth.png)
        * The black line is the Train set y value, which is the actual opening price on the next day, for a total of 1248 days
        * The orange line is the model prediction, which is the actual opening price on the next day, for a total of 1248 days
        * The red line is the model prediction, that is, the model prediction of the next day's opening price, a total of 1248 days
## TESTING RESULT
* Model 1:
  * Model-1 open
    * ![Model-1 open](./img/plot_model1_pred_open.png)
  * Model-1 close
    * ![Model-2 close](./img/plot_model1_pred_close.png)
* Model 2:
  * Model-2 open
    * ![Model-1 open](./img/plot_model2_pred_open.png)
  * Model-2 close
    * ![Model-2 close](./img/plot_model2_pred_close.png)
## EVALUTION
(To be tested)