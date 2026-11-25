
import re
import pickle
from pathlib import Path

import numpy as np
from nlp import tokenize as nlp_tokenize

from training_data import (
    data as DEFAULT_TRAINING_DATA,
    validation_data as DEFAULT_VALIDATION_DATA
)


class NeuralNetwork:

    def __init__(
        self,
        hidden_sizes=(32, 16),
        learning_rate=0.01,
        epochs=500,
        batch_size=8,
        dropout_rate=0.10,
        l2=0.0001,
        patience=40,
        seed=42
    ):

        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.l2 = l2
        self.patience = patience
        self.seed = seed

        self.vocabulary = []
        self.labels = []

        self.weights = []
        self.biases = []

        self.trained = False

    # -------------------------
    # TEXT PROCESSING
    # -------------------------

    def tokenize(self, text):
        return nlp_tokenize(str(text))


    def features(self, text):
        tokens = list(self.tokenize(text))

        bigrams = [
            tokens[i] + "__" + tokens[i + 1]
            for i in range(len(tokens) - 1)
        ]

        return tokens + bigrams

    # -------------------------
    # ENCODING
    # -------------------------

    def encode(self, text):

        items = self.features(text)

        vector = np.array(
            [
                items.count(word)
                for word in self.vocabulary
            ],
            dtype=np.float32
        )

        total = vector.sum()

        if total > 0:
            vector /= total

        return vector

    # -------------------------
    # SOFTMAX
    # -------------------------

    def softmax(self, values):

        values = values - np.max(
            values,
            axis=1,
            keepdims=True
        )

        exp_values = np.exp(values)

        return exp_values / np.sum(
            exp_values,
            axis=1,
            keepdims=True
        )

    # -------------------------
    # INITIALIZE NETWORK
    # -------------------------

    def initialize(self):

        rng = np.random.default_rng(self.seed)

        sizes = [
            len(self.vocabulary),
            *self.hidden_sizes,
            len(self.labels)
        ]

        self.weights = []
        self.biases = []

        for i in range(len(sizes) - 1):

            fan_in = sizes[i]
            fan_out = sizes[i + 1]

            scale = np.sqrt(2 / fan_in)

            weight = rng.standard_normal(
                (fan_in, fan_out)
            ).astype(np.float32) * scale

            bias = np.zeros(
                fan_out,
                dtype=np.float32
            )

            self.weights.append(weight)
            self.biases.append(bias)

    # -------------------------
    # FORWARD PROPAGATION
    # -------------------------

    def forward(self, X, training=False, rng=None):

        activations = [X]
        hidden_values = []
        dropout_masks = []

        current = X

        for i in range(len(self.weights) - 1):

            z = current @ self.weights[i] + self.biases[i]

            hidden = np.tanh(z)

            hidden_values.append(hidden)

            if training and self.dropout_rate > 0:

                mask = (
                    rng.random(hidden.shape)
                    > self.dropout_rate
                ).astype(np.float32)

                mask /= (1 - self.dropout_rate)

                hidden = hidden * mask

            else:
                mask = np.ones_like(hidden)

            dropout_masks.append(mask)

            current = hidden
            activations.append(current)

        logits = (
            current @ self.weights[-1]
            + self.biases[-1]
        )

        probabilities = self.softmax(logits)

        return (
            probabilities,
            activations,
            hidden_values,
            dropout_masks
        )

    # -------------------------
    # LOSS
    # -------------------------

    def calculate_loss(self, probabilities, targets):

        count = len(targets)

        selected = probabilities[
            np.arange(count),
            targets
        ]

        loss = -np.mean(
            np.log(selected + 1e-9)
        )

        regularization = sum(
            np.sum(weight ** 2)
            for weight in self.weights
        )

        return loss + self.l2 * regularization / 2

    # -------------------------
    # VALIDATION
    # -------------------------

    def evaluate_arrays(self, X, y):

        probabilities, _, _, _ = self.forward(X)

        loss = self.calculate_loss(
            probabilities,
            y
        )

        predictions = np.argmax(
            probabilities,
            axis=1
        )

        accuracy = np.mean(
            predictions == y
        ) * 100

        return loss, accuracy

    # -------------------------
    # TRAINING
    # -------------------------

    def fit(
        self,
        training_data=None,
        validation_data=None
    ):
        if training_data is None:
            training_data = DEFAULT_TRAINING_DATA

        if validation_data is None:
            validation_data = DEFAULT_VALIDATION_DATA

        if not training_data:
            raise ValueError("Training data is empty!")

        if not all(
            isinstance(item, (tuple, list))
            and len(item) == 2
            and str(item[0]).strip()
            and str(item[1]).strip()
            for item in training_data
        ):
            raise ValueError("Invalid training data!")

        self.labels = sorted(
            set(label for _, label in training_data)
        )

        self.vocabulary = sorted(
            set(
                feature
                for sentence, _ in training_data
                for feature in self.features(sentence)
            )
        )

        if not self.vocabulary:
            raise ValueError("Vocabulary is empty!")

        X = np.array(
            [self.encode(sentence) for sentence, _ in training_data],
            dtype=np.float32
        )

        y = np.array(
            [
                self.labels.index(label)
                for _, label in training_data
            ],
            dtype=np.int64
        )

        if validation_data:

            for sentence, label in validation_data:

                if label not in self.labels:
                    raise ValueError(
                        f"Unknown validation label: {label}"
                    )

            X_val = np.array(
                [
                    self.encode(sentence)
                    for sentence, _ in validation_data
                ],
                dtype=np.float32
            )

            y_val = np.array(
                [
                    self.labels.index(label)
                    for _, label in validation_data
                ],
                dtype=np.int64
            )

        else:
            X_val = X
            y_val = y

        self.initialize()

        rng = np.random.default_rng(self.seed)

        # Adam optimizer memory

        mW = [np.zeros_like(w) for w in self.weights]
        vW = [np.zeros_like(w) for w in self.weights]

        mb = [np.zeros_like(b) for b in self.biases]
        vb = [np.zeros_like(b) for b in self.biases]

        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8

        step = 0

        best_loss = float("inf")
        best_weights = None
        best_biases = None

        waiting = 0

        print("\nNeural Network Training Started!")

        for epoch in range(self.epochs):

            indices = rng.permutation(len(X))

            for start in range(
                0,
                len(X),
                self.batch_size
            ):

                batch_indices = indices[
                    start:start + self.batch_size
                ]

                batch_X = X[batch_indices]
                batch_y = y[batch_indices]

                (
                    probabilities,
                    activations,
                    hidden_values,
                    masks
                ) = self.forward(
                    batch_X,
                    training=True,
                    rng=rng
                )

                batch_count = len(batch_y)

                gradient = probabilities.copy()

                gradient[
                    np.arange(batch_count),
                    batch_y
                ] -= 1

                gradient /= batch_count

                dW = [None] * len(self.weights)
                db = [None] * len(self.biases)

                for layer in reversed(
                    range(len(self.weights))
                ):

                    dW[layer] = (
                        activations[layer].T @ gradient
                        + self.l2 * self.weights[layer]
                    )

                    db[layer] = gradient.sum(axis=0)

                    if layer > 0:

                        gradient = (
                            gradient @ self.weights[layer].T
                        )

                        gradient *= (
                            1 - hidden_values[layer - 1] ** 2
                        )

                        gradient *= masks[layer - 1]

                # Adam updates

                step += 1

                for i in range(len(self.weights)):

                    mW[i] = (
                        beta1 * mW[i]
                        + (1 - beta1) * dW[i]
                    )

                    vW[i] = (
                        beta2 * vW[i]
                        + (1 - beta2) * (dW[i] ** 2)
                    )

                    mb[i] = (
                        beta1 * mb[i]
                        + (1 - beta1) * db[i]
                    )

                    vb[i] = (
                        beta2 * vb[i]
                        + (1 - beta2) * (db[i] ** 2)
                    )

                    corrected_mW = (
                        mW[i] / (1 - beta1 ** step)
                    )

                    corrected_vW = (
                        vW[i] / (1 - beta2 ** step)
                    )

                    corrected_mb = (
                        mb[i] / (1 - beta1 ** step)
                    )

                    corrected_vb = (
                        vb[i] / (1 - beta2 ** step)
                    )

                    self.weights[i] -= (
                        self.learning_rate
                        * corrected_mW
                        / (np.sqrt(corrected_vW) + epsilon)
                    )

                    self.biases[i] -= (
                        self.learning_rate
                        * corrected_mb
                        / (np.sqrt(corrected_vb) + epsilon)
                    )

            val_loss, val_accuracy = self.evaluate_arrays(
                X_val,
                y_val
            )

            if epoch % 25 == 0:

                print(
                    f"Epoch: {epoch + 1} | "
                    f"Loss: {val_loss:.4f} | "
                    f"Accuracy: {val_accuracy:.2f}%"
                )

            # Early stopping

            if val_loss < best_loss:

                best_loss = val_loss

                best_weights = [
                    weight.copy()
                    for weight in self.weights
                ]

                best_biases = [
                    bias.copy()
                    for bias in self.biases
                ]

                waiting = 0

            else:

                waiting += 1

            if waiting >= self.patience:

                print("\nEarly stopping activated!")
                break

        if best_weights is not None:

            self.weights = best_weights
            self.biases = best_biases

        self.trained = True

        train_loss, train_accuracy = self.evaluate_arrays(
            X,
            y
        )

        val_loss, val_accuracy = self.evaluate_arrays(
            X_val,
            y_val
        )

        print("\nTraining Finished!")
        print(f"Training Accuracy: {train_accuracy:.2f}%")
        print(f"Validation Accuracy: {val_accuracy:.2f}%")

        return {
            "training_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
            "training_loss": train_loss,
            "validation_loss": val_loss
        }

    # -------------------------
    # PREDICTION
    # -------------------------

    def predict(self, text):

        if not self.trained:
            raise RuntimeError("Train or load the model first!")

        vector = self.encode(text).reshape(1, -1)

        if vector.sum() == 0:
            return None, 0.0

        probabilities, _, _, _ = self.forward(vector)

        index = int(np.argmax(probabilities[0]))

        label = self.labels[index]
        confidence = float(probabilities[0][index])

        return label, confidence


        # -------------------------
    # AI BRAIN INTEGRATION
    # -------------------------

    def get_embeddings(self):
        from embeddings import get_trained_embeddings

        return get_trained_embeddings()

    def compare_sentences(self, text1, text2):
        from embeddings import sentence_similarity

        vocabulary, vectors = self.get_embeddings()

        return sentence_similarity(
            text1,
            text2,
            vocabulary,
            vectors
        )

    def search_local_knowledge(self, question):
        from knowledge_search import search_knowledge

        return search_knowledge(question)

    def retrieve_information(self, question):
        from rag import run_rag

        return run_rag(question)

    def remember(self, question, answer):
        from memory import save_memory
        return save_memory(question, answer)    

    
    def understand(self, question):
        # Neural Network prediction
        intent, confidence = self.predict(question)

        # RAG information
        rag_result = self.retrieve_information(question)

        # Previous conversation memory
        from memory import get_memory_context

        memory_context = get_memory_context(
            question,
            limit=5
        )

        return {
            "question": question,
            "intent": intent,
            "confidence": confidence,
            "knowledge": rag_result,
            "found": rag_result.get("found", False),
            "answer": rag_result.get("answer", ""),
            "memory": memory_context
        }

        

    # -------------------------
    # SAVE MODEL
    # -------------------------

    def save(self, path="brain.pkl"):

        if not self.trained:
            raise RuntimeError("Model is not trained!")

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        model_data = {
            "version": 2,
            "hidden_sizes": self.hidden_sizes,
            "vocabulary": self.vocabulary,
            "labels": self.labels,
            "weights": self.weights,
            "biases": self.biases
        }

        with open(path, "wb") as file:
            pickle.dump(model_data, file)

        print("Neural Network saved:", path)

    # -------------------------
    # LOAD MODEL
    # -------------------------

    def load(self, path="brain.pkl"):

        path = Path(path)

        if not path.exists():
            return False

        try:

            with open(path, "rb") as file:
                model_data = pickle.load(file)

            if model_data.get("version") != 2:
                return False

            self.hidden_sizes = tuple(
                model_data["hidden_sizes"]
            )

            self.vocabulary = model_data["vocabulary"]
            self.labels = model_data["labels"]

            self.weights = model_data["weights"]
            self.biases = model_data["biases"]

            self.trained = True

            return True

        except (
            OSError,
            EOFError,
            KeyError,
            ValueError,
            pickle.UnpicklingError
        ):

            return False
