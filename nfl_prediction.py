import requests
import pandas as pd
import joblib
import os
import shutil
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from huggingface_hub import HfApi, Repository

# Full mapping from ESPN location strings to team names
location_to_team = {
    "Arizona": "Cardinals",
    "Atlanta": "Falcons",
    "Baltimore": "Ravens",
    "Buffalo": "Bills",
    "Carolina": "Panthers",
    "Chicago": "Bears",
    "Cincinnati": "Bengals",
    "Cleveland": "Browns",
    "Dallas": "Cowboys",
    "Denver": "Broncos",
    "Detroit": "Lions",
    "Green Bay": "Packers",
    "Houston": "Texans",
    "Indianapolis": "Colts",
    "Jacksonville": "Jaguars",
    "Kansas City": "Chiefs",
    "Las Vegas": "Raiders",
    "Los Angeles": "Rams",        # Note: Charger vs Rams differ by stadium (look for Chargers with "Los Angeles Chargers")
    "Los Angeles Chargers": "Chargers",
    "Miami": "Dolphins",
    "Minnesota": "Vikings",
    "New England": "Patriots",
    "New Orleans": "Saints",
    "New York": "Giants",             # Note: Both Giants and Jets share New York; you may need disambiguation
    "New York Giants": "Giants",
    "New York Jets": "Jets",
    "Philadelphia": "Eagles",
    "Pittsburgh": "Steelers",
    "San Francisco": "49ers",
    "Seattle": "Seahawks",
    "Tampa Bay": "Buccaneers",
    "Tennessee": "Titans",
    "Washington": "Commanders"
}

def fetch_this_weeks_nfl_schedule(year=2025, week=2, seasontype=2):
    """
    Fetch ESPN NFL schedule using the updated JSON API structure,
    extracting home and away team names and game times.
    """
    url = f"https://cdn.espn.com/core/nfl/schedule?xhr=1&year={year}&week={week}&seasontype={seasontype}"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()

    games = []
    schedules = data.get("content", {}).get("schedule", {})
    for day_key in schedules:
        day_sched = schedules[day_key]
        for game in day_sched.get("games", []):
            # Each game has a "competitions" list, usually length 1
            comps = game.get("competitions", [])
            if not comps:
                continue
            comp = comps[0]
            competitors = comp.get("competitors", [])
            home = None
            away = None
            for team in competitors:
                if team.get("homeAway") == "home":
                    home = team.get("team", {}).get("location")
                elif team.get("homeAway") == "away":
                    away = team.get("team", {}).get("location")
            if home and away:
                # Game datetime string, e.g. '2025-09-14T19:00Z'
                game_time = game.get("date", "")
                games.append({"home": home, "away": away, "time": game_time})
    
    # Map locations to full team names for consistency
    for game in games:
        game['home'] = location_to_team.get(game['home'], game['home'])
        game['away'] = location_to_team.get(game['away'], game['away'])

    return games

def load_and_transform_game_data(csv_file):
    """
    Load CSV with home/away columns and transform to per-team rows.
    """
    df = pd.read_csv(csv_file)

    # Define columns for home and away teams
    home_cols = ['season', 'week', 'date', 'time_et', 'neutral', 'home',
                 'score_home', 'first_downs_home', 'first_downs_from_passing_home',
                 'first_downs_from_rushing_home', 'first_downs_from_penalty_home',
                 'third_down_comp_home', 'third_down_att_home',
                 'fourth_down_comp_home', 'fourth_down_att_home',
                 'plays_home', 'drives_home', 'yards_home',
                 'pass_comp_home', 'pass_att_home', 'pass_yards_home',
                 'sacks_num_home', 'sacks_yards_home', 'rush_att_home', 'rush_yards_home',
                 'pen_num_home', 'pen_yards_home', 
                 'redzone_comp_home', 'redzone_att_home',
                 'fumbles_home', 'interceptions_home',
                 'def_st_td_home', 'possession_home']

    away_cols = ['season', 'week', 'date', 'time_et', 'neutral', 'away',
                 'score_away', 'first_downs_away', 'first_downs_from_passing_away',
                 'first_downs_from_rushing_away', 'first_downs_from_penalty_away',
                 'third_down_comp_away', 'third_down_att_away',
                 'fourth_down_comp_away', 'fourth_down_att_away',
                 'plays_away', 'drives_away', 'yards_away',
                 'pass_comp_away', 'pass_att_away', 'pass_yards_away',
                 'sacks_num_away', 'sacks_yards_away', 'rush_att_away', 'rush_yards_away',
                 'pen_num_away', 'pen_yards_away',
                 'redzone_comp_away', 'redzone_att_away',
                 'fumbles_away', 'interceptions_away',
                 'def_st_td_away', 'possession_away']
    
    # Extract home team stats
    home_df = df[home_cols].copy()
    home_df.columns = ['season', 'week', 'date', 'time_et', 'neutral', 'team',
                       'score', 'first_downs', 'first_downs_from_passing', 'first_downs_from_rushing',
                       'first_downs_from_penalty', 'third_down_comp', 'third_down_att',
                       'fourth_down_comp', 'fourth_down_att', 'plays', 'drives', 'yards',
                       'pass_comp', 'pass_att', 'pass_yards', 'sacks_num', 'sacks_yards',
                       'rush_att', 'rush_yards', 'pen_num', 'pen_yards',
                       'redzone_comp', 'redzone_att', 'fumbles', 'interceptions',
                       'def_st_td', 'possession']

    # Extract away team stats
    away_df = df[away_cols].copy()
    away_df.columns = ['season', 'week', 'date', 'time_et', 'neutral', 'team',
                       'score', 'first_downs', 'first_downs_from_passing', 'first_downs_from_rushing',
                       'first_downs_from_penalty', 'third_down_comp', 'third_down_att',
                       'fourth_down_comp', 'fourth_down_att', 'plays', 'drives', 'yards',
                       'pass_comp', 'pass_att', 'pass_yards', 'sacks_num', 'sacks_yards',
                       'rush_att', 'rush_yards', 'pen_num', 'pen_yards',
                       'redzone_comp', 'redzone_att', 'fumbles', 'interceptions',
                       'def_st_td', 'possession']

    # Combine home and away team data
    teams_df = pd.concat([home_df, away_df], ignore_index=True)

    return teams_df

def prepare_features(game, teams_stats_df, le):
    """
    Prepare features for prediction using team's aggregated stats.
    """
    home = game["home"]
    away = game["away"]

    # Aggregate or average stats per team over available data
    home_stats = teams_stats_df[teams_stats_df["team"] == home].mean(numeric_only=True)
    away_stats = teams_stats_df[teams_stats_df["team"] == away].mean(numeric_only=True)

    if home_stats.empty or away_stats.empty:
        raise ValueError("Missing stats for home or away team")

    home_encoded = le.transform([home])[0]
    away_encoded = le.transform([away])[0]

    features = {
        "home_encoded": home_encoded,
        "away_encoded": away_encoded,
        "first_downs_home": home_stats["first_downs"],
        "first_downs_away": away_stats["first_downs"],
        "yards_home": home_stats["yards"],
        "yards_away": away_stats["yards"],
        "pass_comp_home": home_stats["pass_comp"],
        "pass_comp_away": away_stats["pass_comp"],
        "pass_yards_home": home_stats["pass_yards"],
        "pass_yards_away": away_stats["pass_yards"],
        "rush_att_home": home_stats["rush_att"],
        "rush_att_away": away_stats["rush_att"],
        "rush_yards_home": home_stats["rush_yards"],
        "rush_yards_away": away_stats["rush_yards"],
        "pen_num_home": home_stats["pen_num"],
        "pen_num_away": away_stats["pen_num"],
        "pen_yards_home": home_stats["pen_yards"],
        "pen_yards_away": away_stats["pen_yards"],
        "third_down_comp_home": home_stats["third_down_comp"],
        "third_down_comp_away": away_stats["third_down_comp"],
        "third_down_att_home": home_stats["third_down_att"],
        "third_down_att_away": away_stats["third_down_att"],
    }

    return pd.DataFrame([features])

def push_model_to_hub(local_model_path, repo_id, commit_message="Add trained Random Forest model"):
    print(f"Pushing model to Hugging Face Hub repository: {repo_id}")
    api = HfApi()
    try:
        api.create_repo(repo_id, exist_ok=True)
    except Exception as e:
        print(f"Warning on repo creation: {e}")

    repo_local_dir = "hf_" + repo_id.split("/")[-1]
    if not os.path.exists(repo_local_dir):
        repo = Repository(local_dir=repo_local_dir, clone_from=repo_id)
    else:
        repo = Repository(local_dir=repo_local_dir)
        repo.git_pull()

    shutil.copy(local_model_path, repo_local_dir)
    repo.push_to_hub(commit_message=commit_message)
    print("Model pushed successfully.")


def main():
    # Step 1: Fetch this weeks schedule
    print("Fetching this weeks NFL schedule...")
    schedule = fetch_this_weeks_nfl_schedule()
    print(f"Found {len(schedule)} games:")
    for g in schedule:
        print(f"  {g['away']} at {g['home']} @ {g['time']} ET")

    # Step 2: Load historical team stats
    print("\nLoading and transforming historical game stats nfl_team_stats_2002-2024.csv")
    teams_stats_df = load_and_transform_game_data("nfl_team_stats_2002-2024.csv")
    teams = teams_stats_df["team"].unique()
    print(f"Loaded stats for {len(teams)} unique teams.")
    print("Teams: ", teams)
    le = LabelEncoder()
    le.fit(teams)

    # Option 1: Print the entire DataFrame (careful if very large)
    # print(teams_stats_df)

    # Option 2: Print the first few rows to get a snapshot
    # print(teams_stats_df.head())

    # Option 3: Print stats for a specific team (e.g., "Patriots")
    team_name = "Texans"
    team_data = teams_stats_df[teams_stats_df["team"] == team_name]
    print(f"Data for team {team_name}:")
    print(team_data)

    # Option 4: Print all teams with a loop
    # for team in teams_stats_df["team"].unique():
    #     print(f"Stats for {team}:")
    #     print(teams_stats_df[teams_stats_df["team"] == team])
    #     print("-" * 40)
    
    # Prepare training data: features + target (score difference)
    print("Preparing training data...")
    X = []
    y = []

    # Group original csv by game (each row corresponds to one game)
    # We can reuse home/away data per game to create one training example
    for _, row in pd.read_csv("nfl_team_stats_2002-2024.csv").iterrows():
        # Compute score difference target (home - away)
        target = row["score_home"] - row["score_away"]

        # Prepare input features per game using the function
        # We replicate 'game' dict format used in `prepare_features`
        game_dict = {
            "home": row["home"],
            "away": row["away"]
        }

        try:
            feat_df = prepare_features(game_dict, teams_stats_df, le)
        except Exception as e:
            print(f"Skipping game {row['away']}@{row['home']} due to error: {e}")
            continue
        
        X.append(feat_df.iloc[0])
        y.append(target)

    X_df = pd.DataFrame(X)
    print("X_df: ")
    print(X_df)

    y_series = pd.Series(y)
    print("y_series: ")
    print(y_series)

    # Split to train/test sets 80/20
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_series, test_size=0.2, random_state=42)

    # Train-test split
    print("Training Random Forest Regression model...")
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_series, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)

    # Print the True results vs Predictions for comparison from a DataFrame
    results_df = pd.DataFrame({'True': y_test, 'Predicted': preds})
    print(results_df.head())  # shows top few comparisons

    mae = mean_absolute_error(y_test, preds)
    print(f"Validation MAE: {mae:.2f}")

    # # Save model locally
    # model_path = "rf_nfl_model.joblib"
    # joblib.dump(model, model_path)
    # print(f"Model saved locally as {model_path}")
# 
    # # Push model to Hugging Face Hub
    # push_model_to_hub(model_path, "your-username/rf-nfl-score-predictor", "Add trained Random Forest model")

    # Predict today's games
    print("\n=================================================================")
    print("Predictions for today's NFL games:")
    for game in schedule:
        try:
            features_df = prepare_features(game, teams_stats_df, le)
            pred = model.predict(features_df)[0]
            if pred < 0:
                print(f"Away team {game['away']} wins by {-pred:.2f}")
            else:
                print(f"Home team {game['home']} wins by {pred:.2f}")
        except Exception as e:
            print(f"Failed prediction {game['away']} @ {game['home']}: {e}")

if __name__ == "__main__":
    main()