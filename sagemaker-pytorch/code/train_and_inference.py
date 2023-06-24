import os
import json
import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
import shutil

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# logger.addHandler(logging.StreamHandler(sys.stdout))

# pytorch model to predict the rating of user and movie
class ANN(nn.Module):
    def __init__(self, U_dim, M_dim, embed_dim):
        """
        Parameters:
            U_dim: int, total number of users
            M_dim: int, total number of movies
            embed_dim: int, embedding dimension
        """
        super(ANN, self).__init__()
        self.user_embed = nn.Embedding(U_dim, embed_dim)
        self.movie_embed = nn.Embedding(M_dim, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(2 * embed_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def forward(self, X):
        """
        X: two dimensional tensor, the first column is user IDs, and second column is movie IDs
        """
        u_embed = self.user_embed(X[:,0])
        m_embed = self.movie_embed(X[:,1])
        output = torch.cat((u_embed, m_embed), dim=1)
        output = self.fc(output)
        return output
    

def parse_args():
    """
    Parse arguments passed from the SageMaker API
    to the container
    """

    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.1)

    # Data directories
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))

    # Model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))

    return parser.parse_known_args()


def data_generater(X, y, batch_size=32, train=True):
    """
    Parameters:
        X: 2-dimensional np.ndarray, first column is user IDs, and second column is movieIDs
        y: 1-dimensional np.ndarray, rating of a movie for a user
        batch_size: number of samples for one step of training
        train: if for training data generator, shuffle the data
    """
    if train:
        X, y = shuffle(X, y)
    batches = int(np.ceil(len(X) / batch_size))
    for i in range(batches):
        X_batch = X[i * batch_size : (i+1) * batch_size]
        y_batch = y[i * batch_size : (i+1) * batch_size]
        X_batch = torch.from_numpy(X_batch).long()
        y_batch = torch.from_numpy(y_batch).float()
        yield X_batch, y_batch.unsqueeze(-1)


def train(args):
    batch_size = args.batch_size
    epochs = args.epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load train and test data
    train_data = np.load(os.path.join(args.train, "train_data.npy"))
    test_data = np.load(os.path.join(args.test, "test_data.npy"))

    # read the user and movie dimensions for a file
    with open(os.path.join(args.train, "total_users_and_movies.txt")) as f:
        s = f.read()
    U_dim, M_dim = [int(c) for c in s.split(",")]

    print(
        "batch_size = {}, epochs = {}".format(batch_size, epochs)
    )
    print("train data shape: {}, test data shape: {}".format(train_data.shape, test_data.shape))
    print("number of users: {}, number of movies: {}".format(U_dim, M_dim))

    # create model and train the model
    model = ANN(U_dim, M_dim, 10)
    model = model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    train_losses = []
    test_losses = []

    for i in range(epochs):
        start_time = datetime.now()
        train_loss = []
        # train_data has three columns, the first two clumns are user and movie IDs. last column is rating
        for X, y in data_generater(train_data[:,:2], train_data[:,2], batch_size):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        train_losses.append(np.mean(train_loss))

        test_loss = []
        with torch.no_grad():
            for X, y in data_generater(test_data[:,:2], test_data[:,2], batch_size, train=False):
                X, y = X.to(device), y.to(device)
                y_hat = model(X)
                loss = loss_fn(y_hat, y)
                test_loss.append(loss.item())
        test_losses.append(np.mean(test_loss))

        print(
            "epoch: {}, train loss: {}, test loss: {}, run time: {}"\
                .format(i, train_losses[i], test_losses[i], datetime.now()-start_time)
        )

    df = pd.DataFrame({"train_losses":train_losses, "test_losses":test_losses})
    # save model
    torch.save(model.state_dict(), args.model_dir + "/model.pth")
    #save the user and movie dimensions in the model_dir
    np.save(args.model_dir + "/user_movie_dim.npy", np.array([U_dim, M_dim]))
    print("model saved to {}".format(args.model_dir))

    df.to_csv(args.output_data_dir + "/losses.csv")

    script_path = os.path.join(args.model_dir, "code")
    if not os.path.exists(script_path):
        os.mkdir(script_path)
        print("Created a floder at {}!".format(script_path))

    shutil.copy("train_and_inference.py", script_path)
    print("Saving python code to {}.".format(script_path))

def model_fn(model_dir):
    """
    Load the model for inference
    """
    U_dim, M_dim = np.load(model_dir + "/user_movie_dim.npy")
    model = ANN(U_dim, M_dim, 10)
    model.load_state_dict(torch.load(model_dir + "/model.pth"))
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return model.to(device)

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        request = json.loads(request_body)
        return torch.tensor(request).long()

def predict_fn(input_data, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        return model(input_data.to(device)).squeeze(-1)
    

if __name__ == '__main__':
    
    args, _ = parse_args()
    train(args)