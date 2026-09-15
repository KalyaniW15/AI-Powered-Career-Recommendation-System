"""Privacy-safe Matrix Factorization recommendation demo."""

from pathlib import Path
import numpy as np
import pandas as pd


class MatrixFactorizationRecommender:
    """Bias-aware collaborative-filtering model using SGD."""

    def __init__(self, factors=5, learning_rate=0.01,
                 regularization=0.02, epochs=20, random_state=42):
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, interactions):
        required = {'student_id', 'career_id', 'interaction_strength'}
        missing = required.difference(interactions.columns)
        if missing:
            raise ValueError(f'Missing columns: {sorted(missing)}')

        self.interactions = interactions.copy()
        self.student_ids = self.interactions['student_id'].unique()
        self.career_ids = self.interactions['career_id'].unique()

        self.student_to_index = {
            value: index for index, value in enumerate(self.student_ids)
        }
        self.career_to_index = {
            value: index for index, value in enumerate(self.career_ids)
        }

        users = self.interactions['student_id'].map(
            self.student_to_index
        ).to_numpy()
        careers = self.interactions['career_id'].map(
            self.career_to_index
        ).to_numpy()
        scores = self.interactions['interaction_strength'].to_numpy(float)

        generator = np.random.default_rng(self.random_state)
        self.global_mean = scores.mean()
        self.student_bias = np.zeros(len(self.student_ids))
        self.career_bias = np.zeros(len(self.career_ids))
        self.student_factors = generator.normal(
            0, 0.05, (len(self.student_ids), self.factors)
        )
        self.career_factors = generator.normal(
            0, 0.05, (len(self.career_ids), self.factors)
        )

        for _ in range(self.epochs):
            for position in generator.permutation(len(scores)):
                user = users[position]
                career = careers[position]
                actual = scores[position]

                prediction = (
                    self.global_mean
                    + self.student_bias[user]
                    + self.career_bias[career]
                    + np.dot(
                        self.student_factors[user],
                        self.career_factors[career]
                    )
                )

                error = actual - prediction
                old_user = self.student_factors[user].copy()
                old_career = self.career_factors[career].copy()

                self.student_bias[user] += self.learning_rate * (
                    error - self.regularization * self.student_bias[user]
                )
                self.career_bias[career] += self.learning_rate * (
                    error - self.regularization * self.career_bias[career]
                )
                self.student_factors[user] += self.learning_rate * (
                    error * old_career - self.regularization * old_user
                )
                self.career_factors[career] += self.learning_rate * (
                    error * old_user - self.regularization * old_career
                )

        return self

    def predict(self, student_id, career_id):
        user = self.student_to_index[student_id]
        career = self.career_to_index[career_id]
        prediction = (
            self.global_mean
            + self.student_bias[user]
            + self.career_bias[career]
            + np.dot(
                self.student_factors[user],
                self.career_factors[career]
            )
        )
        return float(np.clip(prediction, 0, 1))

    def recommend(self, student_id, top_n=5):
        observed = set(
            self.interactions.loc[
                self.interactions['student_id'] == student_id,
                'career_id'
            ]
        )

        candidates = [
            career_id for career_id in self.career_ids
            if career_id not in observed
        ]

        results = pd.DataFrame({
            'career_id': candidates,
            'predicted_score': [
                self.predict(student_id, career_id)
                for career_id in candidates
            ]
        })

        return results.sort_values(
            'predicted_score', ascending=False
        ).head(top_n).reset_index(drop=True)


if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[1]
    sample_path = project_root / 'data' / 'sample_interactions.csv'
    sample_data = pd.read_csv(sample_path)

    model = MatrixFactorizationRecommender(epochs=20)
    model.fit(sample_data)

    sample_student = sample_data['student_id'].iloc[0]
    print(f'Top recommendations for {sample_student}:')
    print(model.recommend(sample_student, top_n=5))