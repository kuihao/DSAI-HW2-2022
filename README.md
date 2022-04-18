# DSAI-HW2-2022
The homework-2 of the [NCKU](https://www.ncku.edu.tw/index.php?Lang=en) course which named Competitions in [**D**ata **S**ciences and **A**rtificial **I**ntelligence](http://class-qry.acad.ncku.edu.tw/syllabus/online_display.php?syear=0110&sem=2&co_no=P75J000&class_code=).<br>
[The example code](https://github.com/NCKU-CCS/DSAI-HW2-2021).
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
### Q&A
Can I buy one stock at first day (status = 1), the sell and short selling of it at the same sencond day (status = -1)?  
## NSTRUCTIONS FOR USE
(環境建置及程式使用說明)
### Prerequisite
- [conda](https://docs.conda.io/en/latest/index.html)
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
(lstm)
## LISTING 
(explain the function of each file)
## ENVIRONMENT AND EXPERIMENT DESIGN
* Python: 3.6 (3.6.4)
* OS: Ubuntu 20.04 amd64
* GPU: Nvidia Geforce GTX 1070
* [Dataset](https://www.nasdaq.com/market-activity/stocks/ibm), [Training Data](https://www.dropbox.com/s/uwift61i6ca9g3w/training.csv?dl=0), [Testing Data](https://www.dropbox.com/s/duqiffdpcadu6s7/testing.csv?dl=0)
* Random seed:
## TRAINING AND TUNING
### TRAINING RESULT
## EVALUTION