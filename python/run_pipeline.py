# run_pipeline.py
import subprocess

print("Step 1: Building model...")
subprocess.run(["python3", "python/build_model.py"], check=True)

print("Step 2: Predicting upcoming matchups...")
subprocess.run(["python3", "python/predict_matchups.py"], check=True)

print("Step 3: Generating HTML site...")
subprocess.run(["python3", "python/pretty.py"], check=True)

print("Pipeline complete!")
