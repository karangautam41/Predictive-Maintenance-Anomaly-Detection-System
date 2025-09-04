# =============================================================================
# 1. IMPORTS
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from sklearn.preprocessing import MinMaxScaler
import joblib
import keras_tuner as kt
import os
import plotly.graph_objects as go

# =============================================================================
# 2. CONFIGURATION
# =============================================================================
st.set_page_config(page_title="Predictive Maintenance AI", layout="wide")

# Constants for data generation and model
TIME_STEPS = 12
N_FEATURES = 3
MODEL_PATH = 'final_model.h5'
SCALER_PATH = 'scaler.pkl'
THRESHOLD_PATH = 'threshold.pkl'

# =============================================================================
# 3. MACHINE LEARNING HELPER FUNCTIONS (The Backend Logic)
# =============================================================================

def simulate_multivariate_data():
    """Simulates a multivariate time series for N_FEATURES sensors."""
    np.random.seed(42)
    time = np.arange(0, 800, 1)
    temp = 25 + 5 * np.sin(time / 40) + np.random.normal(0, 0.5, len(time))
    vibration = 1.5 + np.random.normal(0, 0.1, len(time))
    pressure = 50 + 0.02 * time + np.random.normal(0, 0.3, len(time))
    data_df = pd.DataFrame({'temperature': temp, 'vibration': vibration, 'pressure': pressure})

    # Inject anomalies
    data_df.loc[250:260, 'temperature'] += 10
    data_df.loc[250:260, 'vibration'] += 0.5
    data_df.loc[600:620, 'pressure'] -= 10
    data_df.loc[600:620, 'vibration'] -= 0.3

    return data_df

def preprocess_data(df):
    """Scales data and creates sequences for LSTM."""
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df)
    train_data = data_scaled[:200]

    def create_sequences(data, n_steps):
        X = []
        for i in range(len(data) - n_steps + 1):
            X.append(data[i:(i + n_steps)])
        return np.array(X)

    X_train = create_sequences(train_data, TIME_STEPS)
    return X_train, scaler

def build_model_for_tuning(hp):
    """Builds a tunable LSTM autoencoder model."""
    input_layer = Input(shape=(TIME_STEPS, N_FEATURES))
    lstm_units = hp.Int('lstm_units', min_value=32, max_value=128, step=32)
    encoder = LSTM(lstm_units, activation='relu')(input_layer)
    repeater = RepeatVector(TIME_STEPS)(encoder)
    decoder = LSTM(lstm_units, activation='relu', return_sequences=True)(repeater)
    output_layer = TimeDistributed(Dense(N_FEATURES))(decoder)
    model = Model(inputs=input_layer, outputs=output_layer)
    lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mae')
    return model

@st.cache_resource
def train_full_model():
    """Executes the complete model training pipeline."""
    data_df = simulate_multivariate_data()
    X_train, scaler = preprocess_data(data_df)
    
    tuner = kt.Hyperband(
        build_model_for_tuning,
        objective='val_loss',
        max_epochs=20,
        factor=3,
        directory='keras_tuner_dir',
        project_name='predictive_maintenance'
    )
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    tuner.search(X_train, X_train, epochs=50, validation_split=0.2, callbacks=[stop_early], verbose=0)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hps)

    history = best_model.fit(
        X_train, X_train,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min')],
        verbose=0
    )

    train_pred = best_model.predict(X_train, verbose=0)
    train_mae_loss = np.mean(np.abs(train_pred - X_train), axis=(1, 2))
    threshold = np.mean(train_mae_loss) + 3 * np.std(train_mae_loss)

    best_model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(threshold, THRESHOLD_PATH)

    return best_hps, history

def predict_anomalies(data_df):
    """Loads artifacts and makes predictions on the uploaded data."""
    if not all([os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, THRESHOLD_PATH]]):
        return None, "Model artifacts not found! Please train the model first in 'Training Mode'."

    model = load_model(MODEL_PATH, compile=False)
    model.compile(optimizer='adam', loss='mae')
    scaler = joblib.load(SCALER_PATH)
    threshold = joblib.load(THRESHOLD_PATH)

    scaled_data = scaler.transform(data_df)
    sequences = [scaled_data[i:(i + TIME_STEPS)] for i in range(len(scaled_data) - TIME_STEPS + 1)]
    X_test = np.array(sequences)

    if X_test.shape[0] == 0:
        return None, "Not enough data to create a full sequence. Please upload more data."

    predictions = model.predict(X_test, verbose=0)
    loss = np.mean(np.abs(predictions - X_test), axis=(1, 2))

    loss_padded = np.full(len(data_df), np.nan)
    loss_padded[TIME_STEPS - 1:] = loss
    results_df = data_df.copy()
    results_df['loss'] = loss_padded
    results_df['threshold'] = threshold
    results_df['anomaly'] = results_df['loss'] > results_df['threshold']

    return results_df, None

def plot_interactive_results(results_df):
    """Creates interactive Plotly charts for each sensor."""
    charts = []
    for col in results_df.columns[:N_FEATURES]:
        fig = go.Figure()
        # Plot normal sensor readings
        fig.add_trace(go.Scatter(x=results_df.index, y=results_df[col], mode='lines', name=f'Normal {col.title()}'))
        # Plot anomalous readings as red markers
        anomalies = results_df[results_df['anomaly']]
        fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies[col], mode='markers', name='Anomaly', marker=dict(color='red', size=8)))
        
        fig.update_layout(
            title=f'Anomalies in {col.title()}',
            xaxis_title='Time Step',
            yaxis_title='Sensor Reading',
            legend_title='Legend'
        )
        charts.append(fig)
    return charts

# =============================================================================
# 4. STREAMLIT UI (The Frontend)
# =============================================================================
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the mode", ["🔎 Prediction Mode", "🚀 Training Mode"])

# --- Training Mode ---
if app_mode == "🚀 Training Mode":
    st.title("🚀 Model Training")
    st.markdown("Train the LSTM Autoencoder to learn normal machine behavior. The model, scaler, and anomaly threshold will be saved for use in Prediction Mode.")

    if st.button("Start Training and Save Model"):
        with st.spinner('Training in progress... This may take several minutes.'):
            best_hps, history = train_full_model()
            st.success("✅ Model training complete and artifacts saved!")

            st.subheader("Training Results")
            col1, col2 = st.columns(2)
            col1.metric("Best LSTM Units Found", best_hps.get('lstm_units'))
            col2.metric("Best Learning Rate Found", f"{best_hps.get('lr'):.4f}")

            st.write("#### Training & Validation Loss Over Epochs")
            loss_df = pd.DataFrame(history.history)
            st.line_chart(loss_df[['loss', 'val_loss']])

# --- Prediction Mode ---
elif app_mode == "🔎 Prediction Mode":
    st.title("🔎 Anomaly Detection for Predictive Maintenance")
    st.markdown("Upload your multivariate sensor data as a CSV file. The AI will analyze it to detect potential machine failures.")

    # Provide a sample file for download
    sample_df = simulate_multivariate_data()
    st.download_button(
        label="Download Sample Data CSV",
        data=sample_df.to_csv(index=False).encode('utf-8'),
        file_name='sample_sensor_data.csv',
        mime='text/csv',
    )

    uploaded_file = st.file_uploader("Choose your CSV file", type="csv")

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        
        if st.button("Analyze for Anomalies"):
            with st.spinner('🧠 Analyzing data...'):
                if input_df.shape[1] != N_FEATURES:
                    st.error(f"The uploaded CSV must have {N_FEATURES} columns. Your file has {input_df.shape[1]}.")
                else:
                    results, error_msg = predict_anomalies(input_df)

                    if error_msg:
                        st.error(error_msg)
                    else:
                        st.success("✅ Analysis complete!")
                        st.header("Analysis Results")

                        # Display anomaly counts for each sensor
                        st.subheader("Anomaly Summary")
                        total_anomalies = results['anomaly'].sum()
                        st.metric("Total Anomalous Timestamps Detected", total_anomalies)

                        # Create interactive plots
                        st.subheader("Interactive Sensor Anomaly Visualization")
                        interactive_charts = plot_interactive_results(results)
                        for chart in interactive_charts:
                            st.plotly_chart(chart, use_container_width=True)

                        # Display reconstruction error chart
                        st.subheader("Reconstruction Error vs. Threshold")
                        error_fig = go.Figure()
                        error_fig.add_trace(go.Scatter(x=results.index, y=results['loss'], mode='lines', name='Reconstruction Error'))
                        error_fig.add_trace(go.Scatter(x=results.index, y=results['threshold'], mode='lines', name='Threshold', line=dict(dash='dash', color='red')))
                        st.plotly_chart(error_fig, use_container_width=True)

                        st.subheader("Data with Anomaly Flags")
                        st.dataframe(results)