# [Previous imports and setup code remains the same...]

class ConstrainedRandomForest(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', bootstrap=True,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state
        )
        self.accommodation_weights = {
            'Hostel': 0.8, 'Airbnb': 0.95, 'Hotel': 1.0,
            'Resort': 1.3, 'Villa': 1.2, 'Guesthouse': 0.85,
            'Vacation rental': 0.9
        }
        self.duration_multiplier = 0.05  # 5% increase per day

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        base_pred = self.model.predict(X)
        
        if isinstance(X, pd.DataFrame):
            if 'AccommodationType' in X.columns:
                acc_types = X['AccommodationType']
                acc_multipliers = acc_types.map(lambda x: self.accommodation_weights.get(x, 1.0)).values
                base_pred = base_pred * acc_multipliers
            
            if 'Duration' in X.columns:
                # New proportional duration calculation
                base_daily_cost = base_pred / 7  # 7 days is our baseline
                base_pred = base_daily_cost * X['Duration']
                duration_effect = 1 + (X['Duration'] * self.duration_multiplier)
                base_pred = base_pred * duration_effect
        
        return base_pred

# [Rest of the code remains the same...]
