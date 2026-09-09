"""
    This script generates a synthetic dataset for training a neural network to detect the presence of life (e.g., survivors) 
    in a search and rescue scenario using drone sensor data. The dataset includes various environmental features, sensor readings, 
    and a target variable indicating whether life is present.

    This script was mostly AI-generated with Gemini 3.1 Pro
"""

from random import random

import pandas as pd
import numpy as np

# Number of rows you want in your mock dataset
NUM_SAMPLES = 1000

# Metadata (The coordinates you will NOT pass to the neural network later)
latitudes = np.random.uniform(low=-90.0, high=90.0, size=NUM_SAMPLES)
longitudes = np.random.uniform(low=-180.0, high=180.0, size=NUM_SAMPLES)

# Environmental Base Features
# We will create an altitude above terrain metric. 
# Lower altitude = clearer readings for thermal/audio.
altitude_above_terrain = np.random.uniform(low=10.0, high=500.0, size=NUM_SAMPLES)

# Particulates and Gases (adding some noise)
particulate_smoke = np.random.uniform(low=0.0, high=100.0, size=NUM_SAMPLES)
particulate_watervapor = np.random.uniform(low=0.0, high=100.0, size=NUM_SAMPLES)
# Baseline CO2 is roughly 400-420 ppm, we will vary it slightly
co2 = np.random.normal(loc=415, scale=20, size=NUM_SAMPLES)
# Baseline O2 is ~21%, measured here as a percentage
o2 = np.random.normal(loc=21.0, scale=0.5, size=NUM_SAMPLES)


# -------------------------------------------------------------
# GENERATING THE TARGET AND CORRELATED SENSORS
# -------------------------------------------------------------

# We need the model to learn that Life (Target = 1) is correlated with:
# 1. Higher thermal readings
# 2. Higher audio decibels
# 3. Both of those readings are amplified when altitude is lower (Inverse Square Law)

target_present = np.random.choice([0, 1], size=NUM_SAMPLES, p=[0.5, 0.5]) # 30% chance of finding life

thermal_reading = []
audio_reading = []
cell_signal = []
bluetooth_signal = []

for i in range(NUM_SAMPLES):
    # Baseline environment without life
    base_temp = np.random.normal(loc=15.0, scale=10.0) # ambient temp in C
    base_audio = np.random.normal(loc=30.0, scale=5.0) # ambient wind/nature in dB
    
    if target_present[i] == 1:
        # Life is present. 
        # A human is ~37C. The higher the drone, the more that signature blends with ambient.
        # We will create a simplified correlation: 
        # Thermal reading = Ambient Temp + (Human Body Heat / (Altitude/50))
        temp_spike = 37.0 / max(1, (altitude_above_terrain[i] / 50)) 
        thermal_reading.append(base_temp + temp_spike)
        
        # Audio spikes from shouts/movement, degrading with altitude
        audio_spike = np.random.uniform(40, 90) / max(1, (altitude_above_terrain[i] / 100))
        audio_reading.append(base_audio + audio_spike)
        
        # Slight CO2 spike near living things (exhalation)
        co2[i] += np.random.uniform(5, 15)
        rng = np.random.default_rng()

        # 70% chance the survivor still has their devices active
        if rng.random() < 0.70:
            cell_signal.append(round(np.random.uniform(-75.0, -30.0), 2))
            bluetooth_signal.append(round(np.random.uniform(-80.0, -40.0), 2))
        else:
            # Survivor is present, but devices are lost, broken, or dead
            cell_signal.append(round(np.random.uniform(-120.0, -90.0), 2))
            bluetooth_signal.append(round(np.random.uniform(-110.0, -95.0), 2))
        
    else:
        # No life present
        thermal_reading.append(base_temp)
        audio_reading.append(base_audio)

        # Empty terrain: No devices nearby, just faint background RF static
        cell_signal.append(round(np.random.uniform(-120.0, -90.0), 2))
        bluetooth_signal.append(round(np.random.uniform(-110.0, -95.0), 2))
        
        # Add some false positives! 
        # e.g., A hot rock might spike the thermal sensor, but won't spike the audio or CO2
        if np.random.random() > 0.9: 
            thermal_reading[i] += np.random.uniform(10, 20) 


# -------------------------------------------------------------
# BUILD THE DATAFRAME
# -------------------------------------------------------------

data = {
    "latitude": latitudes,
    "longitude": longitudes,
    "altitude-terrain-above-sea-level": np.random.uniform(low=0.0, high=3000.0, size=NUM_SAMPLES), # Just adding raw terrain elevation
    "altitude-drone": altitude_above_terrain,
    "thermal-Reading-Celsius": thermal_reading,
    "audio-Reading-Decibels": audio_reading,
    "particulate-smoke": particulate_smoke,
    "particulate-watervapor": particulate_watervapor,
    "co2": co2,
    "o2": o2,
    "cell-signal-strength": cell_signal,
    "bluetooth-signal-strength": bluetooth_signal,
    "target_present": target_present
}

df = pd.DataFrame(data)

# Save the full CSV for reference, and print the shape of the model-ready data
df.to_csv("./data/training/drone_sar_synthetic_data.csv", index=False)
print(f"Full dataset saved with shape: {df.shape}")