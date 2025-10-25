# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 21:46:27 2019

@author: PRATYUSH, Rahul, Somya, Abhay
"""

from flask import Flask, render_template
from flask_cors import CORS, cross_origin
import numpy as np
import pandas as pd
from datetime import datetime
import crops
import random

# ---------- App Setup ----------
app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app, resources={r"/ticker/*": {"origins": "http://localhost:3000"}})

# ---------- Global Data ----------
commodity_dict = {
    "arhar": "static/Arhar.csv",
    "bajra": "static/Bajra.csv",
    "barley": "static/Barley.csv",
    "copra": "static/Copra.csv",
    "cotton": "static/Cotton.csv",
    "sesamum": "static/Sesamum.csv",
    "gram": "static/Gram.csv",
    "groundnut": "static/Groundnut.csv",
    "jowar": "static/Jowar.csv",
    "maize": "static/Maize.csv",
    "masoor": "static/Masoor.csv",
    "moong": "static/Moong.csv",
    "niger": "static/Niger.csv",
    "paddy": "static/Paddy.csv",
    "ragi": "static/Ragi.csv",
    "rape": "static/Rape.csv",
    "jute": "static/Jute.csv",
    "safflower": "static/Safflower.csv",
    "soyabean": "static/Soyabean.csv",
    "sugarcane": "static/Sugarcane.csv",
    "sunflower": "static/Sunflower.csv",
    "urad": "static/Urad.csv",
    "wheat": "static/Wheat.csv"
}

annual_rainfall = [29, 21, 37.5, 30.7, 52.6, 150, 299, 251.7, 179.2, 70.5, 39.8, 10.9]

base = {
    "Paddy": 1245.5, "Arhar": 3200, "Bajra": 1175, "Barley": 980, "Copra": 5100, "Cotton": 3600,
    "Sesamum": 4200, "Gram": 2800, "Groundnut": 3700, "Jowar": 1520, "Maize": 1175, "Masoor": 2800,
    "Moong": 3500, "Niger": 3500, "Ragi": 1500, "Rape": 2500, "Jute": 1675, "Safflower": 2500,
    "Soyabean": 2200, "Sugarcane": 2250, "Sunflower": 3700, "Urad": 4300, "Wheat": 1350
}

commodity_list = []

# ---------- Commodity Class ----------
class Commodity:
    def __init__(self, csv_name):
        self.name = csv_name
        dataset = pd.read_csv(csv_name)
        self.X = dataset.iloc[:, :-1].values
        self.Y = dataset.iloc[:, 3].values

        from sklearn.tree import DecisionTreeRegressor
        depth = random.randrange(7, 18)
        self.regressor = DecisionTreeRegressor(max_depth=depth)
        self.regressor.fit(self.X, self.Y)

    def getPredictedValue(self, value):
        if value[1] >= 2019:
            fsa = np.array(value).reshape(1, 3)
            return self.regressor.predict(fsa)[0]
        else:
            c = self.X[:, 0:2]
            x = [i.tolist() for i in c]
            fsa = [value[0], value[1]]
            ind = 0
            for i in range(len(x)):
                if x[i] == fsa:
                    ind = i
                    break
            return self.Y[ind]

    def getCropName(self):
        return self.name.split('.')[0]

# ---------- Forecast & Helper Functions ----------
def SixMonthsForecastHelper(name):
    current_month = datetime.now().month
    current_year = datetime.now().year
    current_rainfall = annual_rainfall[current_month - 1]

    name_lower = name.lower()
    commodity = next((c for c in commodity_list if name_lower in c.getCropName().lower()), commodity_list[0])

    month_with_year = []
    for i in range(1, 7):
        if current_month + i <= 12:
            month_with_year.append((current_month + i, current_year, annual_rainfall[current_month + i - 1]))
        else:
            month_with_year.append((current_month + i - 12, current_year + 1, annual_rainfall[(current_month + i - 13) % 12]))

    current_wpi = commodity.getPredictedValue([float(current_month), current_year, current_rainfall])
    crop_price = []
    for m, y, r in month_with_year:
        predicted_value = commodity.getPredictedValue([float(m), y, r])
        change = ((predicted_value - current_wpi) * 100) / current_wpi
        x = datetime(y, m, 1).strftime("%b %y")
        crop_price.append([x, round((predicted_value * base[commodity.getCropName().split('/')[-1].capitalize()]) / 100, 2), round(change, 2)])
    return crop_price

def SixMonthsForecast():
    months_data = [[] for _ in range(6)]
    for commodity in commodity_list:
        crop_forecast = SixMonthsForecastHelper(commodity.getCropName())
        for k, data in enumerate(crop_forecast):
            months_data[k].append((data[1], data[2], commodity.getCropName().split("/")[-1], data[0]))
    for month in months_data:
        month.sort()
    crop_month_wise = []
    for month in months_data:
        crop_month_wise.append([month[0][3], month[-1][2], month[-1][0], month[-1][1], month[0][2], month[0][0], month[0][1]])
    return crop_month_wise

def TopFiveWinners():
    current_month = datetime.now().month
    current_year = datetime.now().year
    current_rainfall = annual_rainfall[current_month - 1]
    prev_month = current_month - 1
    prev_rainfall = annual_rainfall[prev_month - 1]
    current_month_prediction = []
    prev_month_prediction = []
    change = []

    for i in commodity_list:
        current_predict = i.getPredictedValue([float(current_month), current_year, current_rainfall])
        prev_predict = i.getPredictedValue([float(prev_month), current_year, prev_rainfall])
        current_month_prediction.append(current_predict)
        prev_month_prediction.append(prev_predict)
        change.append((((current_predict - prev_predict) * 100 / prev_predict), commodity_list.index(i)))

    sorted_change = sorted(change, reverse=True)
    to_send = []
    for j in range(5):
        perc, idx = sorted_change[j]
        name = commodity_list[idx].getCropName().split('/')[-1].capitalize()
        to_send.append([name, round((current_month_prediction[idx] * base[name]) / 100, 2), round(perc, 2)])
    return to_send

def TopFiveLosers():
    current_month = datetime.now().month
    current_year = datetime.now().year
    current_rainfall = annual_rainfall[current_month - 1]
    prev_month = current_month - 1
    prev_rainfall = annual_rainfall[prev_month - 1]
    current_month_prediction = []
    prev_month_prediction = []
    change = []

    for i in commodity_list:
        current_predict = i.getPredictedValue([float(current_month), current_year, current_rainfall])
        prev_predict = i.getPredictedValue([float(prev_month), current_year, prev_rainfall])
        current_month_prediction.append(current_predict)
        prev_month_prediction.append(prev_predict)
        change.append((((current_predict - prev_predict) * 100 / prev_predict), commodity_list.index(i)))

    sorted_change = sorted(change)
    to_send = []
    for j in range(5):
        perc, idx = sorted_change[j]
        name = commodity_list[idx].getCropName().split('/')[-1].capitalize()
        to_send.append([name, round((current_month_prediction[idx] * base[name]) / 100, 2), round(perc, 2)])
    return to_send

# ---------- Missing Functions ----------
def TwelveMonthsForecast(name):
    current_month = datetime.now().month
    current_year = datetime.now().year
    current_rainfall = annual_rainfall[current_month - 1]
    name_lower = name.lower()
    commodity = next((c for c in commodity_list if name_lower in c.getCropName().lower()), commodity_list[0])

    month_with_year = []
    for i in range(1, 13):
        if current_month + i <= 12:
            month_with_year.append((current_month + i, current_year, annual_rainfall[current_month + i - 1]))
        else:
            month_with_year.append((current_month + i - 12, current_year + 1, annual_rainfall[(current_month + i - 13) % 12]))

    max_index, min_index = 0, 0
    max_value, min_value = 0, 9999
    wpis = []
    current_wpi = commodity.getPredictedValue([float(current_month), current_year, current_rainfall])
    change = []

    for idx, (m, y, r) in enumerate(month_with_year):
        current_predict = commodity.getPredictedValue([float(m), y, r])
        if current_predict > max_value:
            max_value = current_predict
            max_index = idx
        if current_predict < min_value:
            min_value = current_predict
            min_index = idx
        wpis.append(current_predict)
        change.append(((current_predict - current_wpi) * 100) / current_wpi)

    max_month, max_year, _ = month_with_year[max_index]
    min_month, min_year, _ = month_with_year[min_index]
    crop_price = []
    for idx, val in enumerate(wpis):
        m, y, _ = month_with_year[idx]
        x = datetime(y, m, 1).strftime("%b %y")
        crop_price.append([x, round((val * base[commodity.getCropName().split('/')[-1].capitalize()]) / 100, 2), round(change[idx], 2)])

    max_crop = [datetime(max_year, max_month, 1).strftime("%b %y"), round(max_value * base[commodity.getCropName().split('/')[-1].capitalize()] / 100, 2)]
    min_crop = [datetime(min_year, min_month, 1).strftime("%b %y"), round(min_value * base[commodity.getCropName().split('/')[-1].capitalize()] / 100, 2)]
    return max_crop, min_crop, crop_price

def TwelveMonthPrevious(name):
    name_lower = name.lower()
    current_month = datetime.now().month
    current_year = datetime.now().year
    commodity = next((c for c in commodity_list if name_lower in c.getCropName().lower()), commodity_list[0])

    month_with_year = []
    for i in range(1, 13):
        if current_month - i >= 1:
            month_with_year.append((current_month - i, current_year, annual_rainfall[current_month - i - 1]))
        else:
            month_with_year.append((current_month - i + 12, current_year - 1, annual_rainfall[current_month - i + 11]))

    crop_price = []
    for m, y, r in month_with_year:
        val = commodity.getPredictedValue([float(m), 2013, r])
        crop_price.append([datetime(y, m, 1).strftime("%b %y"), round((val * base[commodity.getCropName().split('/')[-1].capitalize()]) / 100, 2)])

    crop_price.reverse()
    return crop_price

def CurrentMonth(name):
    current_month = datetime.now().month
    current_year = datetime.now().year
    current_rainfall = annual_rainfall[current_month - 1]
    name_lower = name.lower()
    commodity = next((c for c in commodity_list if name_lower in c.getCropName().lower()), commodity_list[0])
    current_wpi = commodity.getPredictedValue([float(current_month), current_year, current_rainfall])
    current_price = (base[commodity.getCropName().split('/')[-1].capitalize()] * current_wpi) / 100
    return current_price

# ---------- Routes ----------
@app.route('/')
def index():
    context = {
        "top5": TopFiveWinners(),
        "bottom5": TopFiveLosers(),
        "sixmonths": SixMonthsForecast()
    }
    return render_template('index.html', context=context)

@app.route('/commodity/<name>')
def crop_profile(name):
    max_crop, min_crop, forecast_crop_values = TwelveMonthsForecast(name)
    prev_crop_values = TwelveMonthPrevious(name)
    forecast_x = [i[0] for i in forecast_crop_values]
    forecast_y = [i[1] for i in forecast_crop_values]
    previous_x = [i[0] for i in prev_crop_values]
    previous_y = [i[1] for i in prev_crop_values]
    current_price = CurrentMonth(name)
    crop_data = crops.crop(name)
    context = {
        "name": name,
        "max_crop": max_crop,
        "min_crop": min_crop,
        "forecast_values": forecast_crop_values,
        "forecast_x": str(forecast_x),
        "forecast_y": forecast_y,
        "previous_values": prev_crop_values,
        "previous_x": previous_x,
        "previous_y": previous_y,
        "current_price": current_price,
        "image_url": crop_data[0],
        "prime_loc": crop_data[1],
        "type_c": crop_data[2],
        "export": crop_data[3]
    }
    return render_template('commodity.html', context=context)

@app.route('/ticker/<item>/<number>')
@cross_origin(origin='localhost', headers=['Content- Type', 'Authorization'])
def ticker(item, number):
    n = int(number)
    i = int(item)
    data = SixMonthsForecast()
    context = str(data[n][i])
    if i in [2, 5]:
        context = '₹' + context
    elif i in [3, 6]:
        context = context + '%'
    return context

# ---------- Main ----------
if __name__ == "__main__":
    for key in commodity_dict:
        commodity_list.append(Commodity(commodity_dict[key]))
    app.run(debug=True)
