import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import seaborn as sns


def generate_injuries(injury_ratios, num_games, device):
    injury_distributions = torch.distributions.Geometric(torch.tensor(injury_ratios))
    injury_intervals = injury_distributions.sample(torch.tensor([10])).T.to(device)
    injury_indices = injury_intervals.cumsum(dim=1, dtype=torch.uint8)
    injuries = torch.zeros((len(players), num_games)).to(device)
    for indices, player in zip(injury_indices, injuries):
        for injury in indices:
            try:
                player[injury.item()] = 1
                player[injury.item() + 1] = 1
            except IndexError:
                break
    return injuries


def generate_season(players, injury_ratios, absence_ratios, device, num_games=18, complex_injuries=False):
    attendance = torch.ones((len(players), num_games), dtype=torch.uint8).to(device)
    if complex_injuries:
        injuries = generate_injuries(injury_ratios, num_games, device)
    else:
        injuries = torch.zeros_like(attendance)
        injuries[0, :] = 1
    absence_distributions = torch.distributions.Bernoulli(torch.tensor(absence_ratios))
    absences = absence_distributions.sample(torch.tensor([num_games])).T.to(device)
    attendance[5, -6:] = 0
    attendance = F.relu(attendance - absences - injuries)
    attendance = torch.sum(attendance, dim=0)
    return attendance


if __name__ == "__main__":
    players = ["Joren", "Jurjen", "Zwiep", "Frenk", "Sietse", "Wouter", "Rick", "Boom", "Sim", "Beune"]
    injury_ratios = [0.0100, 0.100, 0.0250, 0.0500, 0.100, 0.100, 0.100, 0.0500, 0.0100, 0.050]
    absence_ratios = [0, 0, 0.15, 0.25, 0, 0, 0, 0.3, 0, 0]

    num_seasons = 10000
    num_games = 18
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    overall_attendance = torch.zeros((num_seasons, num_games), dtype=torch.uint8).to(device)
    start = time.time()
    for i in range(num_seasons):
        overall_attendance[i] = generate_season(players, injury_ratios, absence_ratios, device, num_games=num_games,
                                                complex_injuries=True)
    print("Generated {} seasons in {} seconds".format(num_seasons, time.time() - start))
    average_attendance = torch.mean(overall_attendance.float(), dim=0).cpu()
    sorted_seasons = torch.argsort(torch.mean(overall_attendance.float(), dim=1))
    worst_seasons = torch.mean(overall_attendance[sorted_seasons[:100]].float(), dim=0).cpu()
    best_seasons = torch.mean(overall_attendance[sorted_seasons[-100:]].float(), dim=0).cpu()
    plt.figure()
    plt.plot(range(1, 19), average_attendance, label="Average Season")
    plt.plot(range(1, 19), worst_seasons, label="1% Worst Seasons")
    plt.plot(range(1, 19), best_seasons, label="1% Best Seasons")
    plt.xlim([1, 18])
    plt.ylim([0, 10])
    plt.xlabel("Game number")
    plt.ylabel("Number of players")
    plt.legend()
    plt.show()
    plt.figure()
    sns.histplot(overall_attendance.flatten().cpu(), stat="probability", discrete=True)
    plt.show()
    plt.figure()
    sns.histplot(overall_attendance[:, -6:].flatten().cpu(), stat="probability", discrete=True)
    plt.show()
    plt.figure()
    sns.histplot(overall_attendance[:, :-6].flatten().cpu(), stat="probability", discrete=True)
    plt.show()
