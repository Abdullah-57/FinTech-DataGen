# FinTech DataGen 💰📊

*Complete FinTech Application with React.js, Flask, MongoDB, and Machine Learning*

---

## 🏗️ Tech Stack

### 🖥️ Frontend (User Interface)
* **Framework**: React.js
* **Routing**: React Router DOM
* **HTTP Client**: Axios
* **Styling**: CSS3

### ⚙️ Backend
* **Framework**: Flask (Python)
* **CORS**: Flask-CORS
* **Environment**: python-dotenv

### 🧠 Machine Learning Logic
* **Language**: Python
* **Libraries**: scikit-learn, pandas, numpy
* **Models**: RandomForestRegressor

### 🗄️ Database (Storage Layer)
* **Database**: MongoDB
* **Driver**: PyMongo
* **Connection**: MongoDB Atlas

---

## 📁 Project Structure

```
FinTech-DataGen/
├── frontend/                 # React.js Frontend
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard.js
│   │   │   ├── DataGenerator.js
│   │   │   └── Analytics.js
│   │   ├── App.js
│   │   ├── App.css
│   │   ├── index.js
│   │   └── index.css
│   └── package.json
├── backend/                  # Flask Backend
│   ├── database/
│   │   ├── __init__.py
│   │   └── mongodb.py
│   ├── ml_models/
│   │   ├── __init__.py
│   │   ├── predictor.py
│   │   └── feature_engineering.py
│   ├── app.py
│   └── requirements.txt
├── fintech_data_curator.py   # Original data curator
├── env.example              # Environment variables template
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Node.js (v14 or higher)
- Python 3.8+
- MongoDB Atlas account

### 1️⃣ Environment Setup

Create a `.env` file in the backend directory:
```bash
cp env.example backend/.env
```

Edit the `.env` file with your MongoDB credentials:
```
MONGOURI=your_mongodb_connection_string
PORT=5000
FLASK_ENV=development
FLASK_DEBUG=True
```

### 2️⃣ Backend Setup

```bash
cd backend
pip install -r requirements.txt
python app.py
```

The backend will start on `http://localhost:5000`

### 3️⃣ Frontend Setup

```bash
cd frontend
npm install
npm start
```

The frontend will start on `http://localhost:3000`

---

## 🔧 API Endpoints

### Health Check
- `GET /api/health` - Check backend and database connectivity

### Data Generation
- `POST /api/generate` - Generate financial dataset
- `GET /api/datasets` - Get all datasets
- `GET /api/datasets/<id>` - Get specific dataset

### Analytics & Predictions
- `GET /api/analytics` - Get analytics data
- `POST /api/predict` - Make financial prediction

---

## 🧪 Testing the Application

1. **Start Backend**: `cd backend && python app.py`
2. **Start Frontend**: `cd frontend && npm start`
3. **Open Browser**: Navigate to `http://localhost:3000`
4. **Test Features**:
   - Check system status on Dashboard
   - Generate sample financial data
   - View analytics and predictions

---

## 📊 Features

### ✅ Implemented
- React.js frontend with routing
- Flask backend with REST API
- MongoDB integration
- Machine learning model structure
- Data generation using existing curator
- Health check endpoints
- Basic analytics dashboard

### 🔄 In Development
- Advanced ML model training
- Real-time data visualization
- User authentication
- Data export functionality
- Performance optimization

---

## 🛠️ Development Notes

This is a **boilerplate setup** to test the complete tech stack. The application includes:

- **Frontend**: React components for Dashboard, Data Generator, and Analytics
- **Backend**: Flask API with MongoDB integration
- **ML Models**: Basic structure for financial prediction models
- **Database**: MongoDB connection and data models

The existing `fintech_data_curator.py` is integrated into the backend for data generation.

---

## 📚 Next Steps

1. **Test the setup** by running both frontend and backend
2. **Verify MongoDB connection** through the health check
3. **Generate sample data** using the Data Generator
4. **Implement additional features** as needed

---

## 📄 License

MIT License — free to use and modify with credit.

---

## ✨ Credits

**Author**: Abdullah Daoud  
**Institution**: FAST NUCES, BS Software Engineering

🚀 **Ready to build the future of FinTech!**