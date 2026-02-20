from src.preprocessing import preprocess
from src.feature_engineering import engineer_features
from src.train import train
from src.evaluate import evaluate
from src.ab_testing import ab_test

def main():
    print(" Step 1: Preprocessing data")
    df = preprocess()

    print(" Step 2: Feature engineering")
    df_fe = engineer_features(df)

    print(" Step 3: Training model")
    X_test, y_test = train(df_fe)

    print(" Step 4: Evaluating model")
    evaluate(X_test, y_test)

    print(" Step 5: Running A/B test")
    ab_test()

    print("\n Pipeline executed successfully")

if __name__ == "__main__":
    main()
