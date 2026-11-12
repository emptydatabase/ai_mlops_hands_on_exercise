from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def random_forest_predict(X_train, X_test, y_train, random_state):
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf.fit(X_train, y_train)

    return rf.predict(X_test)


def neural_network_predict(X_train, X_test, y_train, random_state: int):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=random_state))
    ])

    pipeline.fit(X_train, y_train)

    return pipeline.predict(X_test)
