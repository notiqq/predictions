import core
import pandas as pd

def train():
    train_df = pd.read_csv('./data/train.csv')
    model = core.NeuralNetwork()
    model.fit(train_df.drop(['target'],axis=1),train_df['target'])
    model.save()


if __name__ == '__main__':
    train()
    