from utils.data_utils import get_prepared_data

if __name__ == "__main__":
    symbols = ["MES=F", "MNQ=F"]
    dtrain, dval, dtest = get_prepared_data(symbols=symbols, interval="1d", period="20y")
    itrain, ival, itest = get_prepared_data(symbols=symbols, interval="60m", period="2mo")
    dtrain.to_csv("data/processed/dtrain.csv", index=False)
    dval.to_csv("data/processed/dtest.csv", index=False)
    dtest.to_csv("data/processed/dval.csv", index=False)
    itrain.to_csv("data/processed/itrain.csv", index=False)
    ival.to_csv("data/processed/itest.csv", index=False)
    itest.to_csv("data/processed/ival.csv", index=False)
    print("\n✅ Data loaded and features created successfully!")