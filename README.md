# 💹 Real-Time Crypto Market Dashboard

![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Plotly](https://img.shields.io/badge/Visualization-Plotly-green?logo=plotly)
![License](https://img.shields.io/badge/License-MIT-yellow)

🚀 **Live Demo:** [Click here to view the app](https://crypto-dashboard-sc4mvcgcwsaqkpddgumnlw.streamlit.app)  
📂 **GitHub Repo:** [crypto-dashboard](https://github.com/SIDDARTHAREDDY8/crypto-dashboard)  
💼 **LinkedIn:** [Siddartha Reddy Chinthala](https://www.linkedin.com/in/siddarthareddy9)

---

## 📌 Overview
The **Real-Time Crypto Market Dashboard** is an interactive Streamlit application that:
- Fetches **live cryptocurrency data** using the [CoinGecko API](https://www.coingecko.com/en/api)
- Displays **candlestick charts** with SMA, EMA, and Bollinger Bands
- Allows **quick range toggles** (1D, 7D, 30D)
- Compares **two coins** on a normalized performance chart
- Provides **returns & volatility metrics**
- Enables **price alerts** with optional **email notifications**
- Supports **CSV downloads** of OHLC + indicator data

---

## 🖥 Features
### 📊 Dashboard Overview
- **Top coins table** with price, market cap, and 24h change
- Auto-refresh support for live market updates

### 📈 Candlestick + Indicators
- SMA, EMA, and Bollinger Bands overlays
- Quick range buttons (1D / 7D / 30D)

### 🆚 Compare Two Coins
- Independent selectors for Coin A & Coin B
- Custom line colors
- Normalized to **Index=100** at the start
- **Returns & volatility** table
- CSV export

### 🔔 Price Alerts
- Set thresholds for coins
- Optional **email alerts** using SMTP (Gmail App Password)

---

## 🛠 Tech Stack
- **Frontend:** [Streamlit](https://streamlit.io/)
- **Data:** [CoinGecko API](https://www.coingecko.com/en/api)
- **Visualization:** [Plotly](https://plotly.com/python/)
- **Language:** Python 3.11

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/SIDDARTHAREDDY8/crypto-dashboard.git
cd crypto-dashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```


---

## 🔐 Email Alerts Setup

If you want to send email alerts:
1. **Create .streamlit/secrets.toml in the project root:**

```bash
[email]
user = "youraddress@gmail.com"
password = "YOUR_16_CHAR_APP_PASSWORD"
smtp_server = "smtp.gmail.com"
smtp_port = 465
```
2. **For Gmail, enable 2FA and generate an App Password:**
Google Support Guide

---

##
📸 Screenshots

![App Screenshot](https://github.com/SIDDARTHAREDDY8/crypto-dashboard/blob/main/img1.png?raw=true)

![App Screenshot](https://github.com/SIDDARTHAREDDY8/crypto-dashboard/blob/main/img2.png?raw=true)

![App Screenshot](https://github.com/SIDDARTHAREDDY8/crypto-dashboard/blob/main/img3.png?raw=true)

![App Screenshot](https://github.com/SIDDARTHAREDDY8/crypto-dashboard/blob/main/img4.png?raw=true)

![App Screenshot](https://github.com/SIDDARTHAREDDY8/crypto-dashboard/blob/main/img5.png?raw=true)

---

##

**📜 License**
This project is licensed under the MIT License.

---
## 👤 Author

**[Siddartha Reddy Chinthala](https://www.linkedin.com/in/siddarthareddy9)**  
🎓 Master’s in CS | Aspiring Data Scientist  
🔗 Connect with me on [LinkedIn](https://www.linkedin.com/in/siddarthareddy9)

⭐️ Show Some Love
If you like this project, don’t forget to ⭐️ the repo and share it!
