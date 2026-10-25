import pandas as pd
from chefboost import Chefboost as chef

# Load the dataset
df = pd.read_csv("data/play_tennis.csv")
print("Dataset loaded:", df.shape[0], "rows")

# A new day to predict: Outlook, Temp, Humidity, Wind
new_day = ["Sunny", "Cool", "Normal", "Weak"]

for algo in ["ID3", "C4.5", "CART"]:
    print("\n" + "=" * 50)
    print("ALGORITHM:", algo)
    print("=" * 50)

    config = {"algorithm": algo}
    model = chef.fit(df.copy(), config=config, target_label="Decision")

    chef.evaluate(model, df.copy(), target_label="Decision")

    result = chef.predict(model, new_day)
    print("Prediction for", new_day, "->", result)