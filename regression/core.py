import tensorflow as tf
import keras

class NeuralNetwork():
    
    def fit(self, x, y):
        x_train = tf.constant(x.values)
        y_train = tf.constant(y.values)

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(53, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(106, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        self.model.compile(loss=tf.keras.losses.mae, 
                    optimizer=tf.keras.optimizers.SGD(), 
                    metrics=[tf.keras.metrics.RootMeanSquaredError()])

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)


        self.model.fit(x_train, y_train,validation_split=0.1, callbacks=[callback], epochs=50)

    def predict(self, x):
        return self.model.predict(x)

    def save(self):
        self.model.save("./models/nn.keras")

    def load(self):
        self.model = keras.models.load_model("./models/nn.keras")