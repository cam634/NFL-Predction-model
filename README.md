# NFL Game Outcome Prediction Model (TensorFlow)

This project builds an end-to-end machine learning pipeline to **predict NFL game winners** using TensorFlow. The model processes historical team performance data, betting market information, and game context to generate **win probabilities** for upcoming matchups.

---

## 📊 Project Overview

The goal of this project is to leverage machine learning to identify patterns that correlate with NFL team success. Using a neural network built with TensorFlow/Keras, the model learns relationships between input features (team stats, efficiency metrics, odds, etc.) and game outcomes.

**Model Performance:**
- **Test Loss:** 0.625  
- **Test AUC:** **0.723**  
  - AUC (Area Under the ROC Curve) measures how well the model distinguishes winners from losers.  
  - AUC > 0.70 indicates **meaningful predictive signal** beyond randomness.

---

## 🏈 Features Used

The model uses a variety of game and team features, including:

- Offensive stats (yards per game, scoring efficiency)
- Defensive stats (opponent yards allowed, pressure rates)
- Team form trends (recent games)
- Betting market implied probabilities (from US odds)
- Home vs. away indicators

All features are normalized and fed into the neural network for training.

---

## 🧠 Model Architecture

Built using **TensorFlow/Keras**, the model consists of:

- Dense feedforward layers  
- ReLU activation functions  
- Sigmoid output layer for win probability  
- Adam optimizer  
- Binary cross-entropy loss  

TensorFlow automates:

- Weight initialization  
- Matrix multiplications  
- Gradient calculation  
- Backpropagation  
- Optimization over many epochs  

This allows the model to learn the best weights to minimize prediction error.

---

## 🔮 How Predictions Work

For each upcoming matchup:

1. Home and away team features are loaded.
2. The neural network performs matrix multiplications with trained weights.
3. Non-linear activations transform the signal.
4. The output layer returns a **probability between 0 and 1**:
   - > 0.5 → home-team favored  
   - < 0.5 → away-team favored  

The model provides both predictions and confidence levels.

---

## 📈 Evaluation Metric — AUC

**AUC (Area Under the ROC Curve)** evaluates how well the model ranks winners above losers.

- **1.0** → perfect predictions  
- **0.5** → no better than a coin flip  
- **0.723** → strong ranking ability for a sports model  

AUC is preferred over accuracy due to the high parity in NFL games.

---

## 🛠️ Technologies Used

- **TensorFlow / Keras**
- **Pandas / NumPy**
- **Scikit-Learn**

---

## 📤 Model Outputs

The model generates:

- Win probabilities  
- Predicted winners  
- Model confidence scores  
- Exportable DataFrames (HTML, CSV, etc.)  
- Visualized tables for weekly predictions  

---

## 🚀 Future Enhancements

- Add player-level stats (QB efficiency, injuries)
- Experiment with model ensembles or XGBoost

---

## 👤 About This Project

Created as a practical application of:

- Machine learning development  
- TensorFlow modeling  
- Sports analytics  
- Data engineering and pipeline automation  
